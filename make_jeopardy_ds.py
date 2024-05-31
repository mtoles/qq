# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
import torch
from datasets import load_from_disk, Dataset
from oracles import *  # T5_Bool_Oracle
from primary_models import get_m1
from secondary_model import (
    Repeater_Secondary_Model,
    OpenAI_Secondary_Model,
    Gt_Secondary_Model,
    Flan_Secondary_Model,
    Alpaca_Secondary_Model,
)
from utils import set_random_seed
from dataset_utils import bf_filtering, combine_adversarial_ds
from datetime import datetime
from time import sleep
import numpy as np

# from masking import bf_del_sentences, bf_add_sentences, reduce_to_n
from masking import (
    randsentence_dataset,
    randdist_dataset,
)
from pathlib import PurePath
import pandas as pd
import os
from preprocess import get_preprocessed_ds
import re

np.random.seed(42)


class Alpaca_Secondary_Model:
    def __init__(
        self,
        model_name,
        model_path,
        max_length=4096,
        device="cuda",
        precision="bf16",
    ):
        self.model_name = model_name
        self.device = device

        if precision == "int8":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", load_in_8bit=True
            )
        elif precision == "bf16":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.alpaca_template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
        # self.alpaca_template_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        self.device = self.device

    def forward(self, masked_sentence):
        prompt = self.alpaca_template.format(
            instruction="Ask a question that can be answered by the context. Begin with who, what, where, or when.",
            input=masked_sentence,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {
            k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
        }
        tries = 0
        # while tries < 3:
        attempts = []
        while tries < 10:
            with torch.no_grad():
                # set model seed
                torch.manual_seed(tries)
                torch.cuda.manual_seed(tries)
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    # temperature=2.0,
                    do_sample=True,
                )

            jeopardy_q = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
            )
            # extract the portion of jeopardy_q in between the question word and the first question mark
            # otherwise, return None
            re_finds = re.search(
                r"\b(?:who|what|where|when)\b.+\?", jeopardy_q, re.IGNORECASE
            )
            if re_finds:
                jeopardy_q = re_finds.group(0)
                return jeopardy_q
            tries += 1
            attempts.append(jeopardy_q)
        print("failed to generate valid question for masked sentence:")
        print(masked_sentence)
        print("attempts:")
        print(attempts)
        print("returning `None`")
        return None


@click.command()
@click.option(
    "--split", default="validation", help="HotpotQA split {train, validation}"
)
@click.option(
    "--m2_arch", help="secondary model architecture {t5, gpt-3.5-turbo, gpt-4, gt}"
)

@click.option(
    "--downsample_pt_size",
    default=None,
    help="use at most this many examples in validation",
)
@click.option(
    "--ds_shift",
    default=0,
    help="Shift the dataset by this many examples before downsampling. Useful for debugging specific examples.",
)
@click.option("--results_filename", help="path to save results")
@click.option(
    "--save_dir", help="directory to save results to", default="data/jeopardy"
)
@click.option(
    "--gt_subset", flag_value=True, help="filter in only gt examples for m2 comparisons"
)
def main(
    split,
    m1_arch,
    m2_arch,
    max_adversarial_examples,
    downsample_pt_size,
    ds_shift,
    gt_subset,
    results_filename,
    save_dir,
):
    masking_scheme = "randsentence"
    set_random_seed(0)

    if ds_shift:
        assert (
            downsample_pt_size is not None
        ), "There is no reason to shift the dataset without downsampling"
    start = datetime.now()
    ds_masking_scheme = (
        "None" if masking_scheme == "bfdelsentence" else "masking_scheme"
    )
    now = datetime.now().strftime("Y%m%d-%H%M%S")
    if results_filename is None:
        results_filename = f"{m1_arch}-{downsample_pt_size}-{ds_masking_scheme}-{now}"

    # Evaluate the primary model
    # m1 = get_m1(m1_path, m1_arch, pm_eval_batch_size)
    # Receive and prepare the primary task
    metrics = {}

    print("preprocessing...")
    # ds = get_preprocessed_ds("validation", downsample_pt_size)
    assert split in ["train", "validation"]
    assert not (
        split == "train" and gt_subset
    ), "gt subset only works for validation since there are no ground truth examples in the training set"
    ds = get_preprocessed_ds(split)

    # downsample if a downsampling size is provided
    if str(downsample_pt_size) != "None":
        ds = ds.select(range(ds_shift, ds_shift + int(downsample_pt_size)))

    original_raw_dataset_len = len(ds)

    # select and mask examples where the primary
    if masking_scheme == "bfsentence":
        raise NotImplementedError
    elif masking_scheme == "randsentence":
        do_gt = m2_arch == "gt" or gt_subset
        m1 = None
        ds = randsentence_dataset(ds, m1, do_gt)
    elif masking_scheme == "randdistsentence":
        raise NotImplementedError
        ds = randdist_dataset(
            ds, m1, max_adversarial_examples
        )  # set drop thresh to -1 so no filtering happens

    masked_sentences = ds["masked_sentence"]
    # downsample and shift
    print("loading alpaca model...")
    alpaca = Alpaca_Secondary_Model(
        "alpaca",
        ".model_cache/alpaca/tuned",
    )
    jeopardy_qs = []
    print("generating jeopardy questions...")
    for i in tqdm(range(0, len(ds))):
        batch = ds[i : i + 1]
        masked_sentences = batch["masked_sentence"]
        batch_output = alpaca.forward(masked_sentences)
        jeopardy_qs.append(batch_output)

    ds = ds.add_column("jeopardy_q", jeopardy_qs)

    # keep only necessary cols
    # q1s = ds["prepped_bfdelsentence_None"][0].split("\n\n")[0]

    # ds.to_hdf("results/jeopardy/jeopardy_ds.hd5", key="ds")

    print

    # # Analysis
    df = pd.DataFrame(ds)
    # print(f"runtime: {datetime.now()-start}")

    # make the dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(
        save_dir,
        f"jeopardy_gpt_{'full' if str(downsample_pt_size) == 'None' else downsample_pt_size}_{split}.jsonl",
    )

    # df.to_hdf(save_path, "ds")
    df.to_json(save_path, orient="records", lines=True)
    print(f"dataset saved to {save_path}")

    print


if __name__ == "__main__":
    main()

# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
import torch
from oracles import *  # T5_Bool_Oracle
from secondary_model import Alpaca_Secondary_Model_Jeopardy_Lookup
from main import main as analyze

from utils import set_random_seed
from datetime import datetime
import numpy as np

# from masking import bf_del_sentences, bf_add_sentences, reduce_to_n
from masking import (
    randsentence_dataset,
)
from pathlib import PurePath
import pandas as pd
import os
from preprocess import get_preprocessed_ds
import re
import json
from nltk.tokenize import sent_tokenize

np.random.seed(42)


def fit_template(q1, context):
    prompt_id = "p1"

    input_template = "Question:\n{q1}\nContext:\n{context}"
    instruction_prefix = (
        "What question can you ask to help you answer the final question:"  # p3
    )
    inpt = input_template.format(context=context, q1=q1)
    instruction = f"{instruction_prefix}\n\n{inpt}"
    # prompt = self.alpaca_template.format(instruction=instruction, input=inpt)

    return instruction


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

    def forward(self, batch):
        masked_sentence = batch["masked_sentence"][0]
        q1 = batch["q1"][0]
        # inpt = f"Task:\n\n{q1}\n\nContext:\n\n{masked_sentence}"
        # prompt = self.alpaca_template.format(
        #     instruction="Ask a question can be answered by the context and will help fill in missing information when answering the task. Begin with who, what, where, or when.",
        #     input=inpt,
        # )
        instruction = f"Context:\n\n{masked_sentence}\n\nAsk a question can be answered by the context. Begin with who, what, where, or when."
        inpt = ""
        prompt = self.alpaca_template.format(instruction=instruction, input=inpt)

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
            # sentences = [s for s in sent_tokenize(jeopardy_q) if s[-1] == "?"]
            # if sentences:
            #     return sentences[-1]
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
@click.option(
    "--save_dir", help="directory to save results to", default="data/jeopardy"
)
@click.option(
    "--gt_subset", flag_value=True, help="filter in only gt examples for m2 comparisons"
)
@click.option(
    "--active_filter",
    flag_value=True,
    help="only include examples where the cq is useful",
)
def main(
    split,
    m2_arch,
    downsample_pt_size,
    ds_shift,
    gt_subset,
    save_dir,
    active_filter,
):
    masking_scheme = "randsentence"
    set_random_seed(0)

    if ds_shift:
        assert (
            downsample_pt_size is not None
        ), "There is no reason to shift the dataset without downsampling"
    start = datetime.now()
    now = datetime.now().strftime("Y%m%d-%H%M%S")

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
        # masked_sentences = batch["masked_sentence"]
        # batch_output = alpaca.forward(masked_sentences)
        batch_output = alpaca.forward(batch)
        jeopardy_qs.append(batch_output)
    del alpaca
    ds = ds.add_column("jeopardy_q", jeopardy_qs)

    # # Analysis
    df = pd.DataFrame(ds)
    # print(f"runtime: {datetime.now()-start}")

    # make the dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(
        save_dir,
        f"jeopardy_{'full' if str(downsample_pt_size) == 'None' else downsample_pt_size}_{split}.jsonl",
    )
    df.to_json(save_path, orient="records", lines=True)

    if active_filter:
        analyze_df = analyze(
            split=split,
            m1_arch="t5-base",
            m2_arch="alexpaca_precomputed",
            alexpaca_precomputed_data_path=save_path,
            oracle_arch="t5",
            oracle_size="base",
            save_dir="results/tmp",
            oracle_eval_batch_size=64,
            m1_eval_batch_size=64,
            # defaults
            alexpaca_path=None,
            template_id=None,
            m2_eval_batch_size=1,
            # max_adversarial_examples,
            downsample_pt_size=downsample_pt_size,
            ds_shift=ds_shift,
            oai_cache_path=None,
            gt_subset=False,
            results_filename="RESULTS_FILENAME",
        )

        # df = df[analyze_df["a2_is_correct"]]
        df = df[analyze_df["m1_masked_a2_f1"] - analyze_df["m1_masked_None_f1"] >= 0.5]
    filtered_save_path = os.path.join(
        save_dir,
        f"jeopardy_{'full' if str(downsample_pt_size) == 'None' else downsample_pt_size}_{split}{'_active_filtered' if active_filter else ''}_tatsu.jsonl",
    )
    # df.to_json(filtered_save_path, orient="records", lines=True)

    df["instruction"] = df.apply(
        lambda x: fit_template(x["q1"], x["fc_masked"]), axis=1
    )
    df["output"] = df["jeopardy_q"]
    df["input"] = ""

    # write instructions, output, and input to a jsonl file as a list of dicts
    output_list = df[["instruction", "input", "output"]].to_dict(orient="records")
    with open(filtered_save_path, "w") as f:
        f.write("[")
        f.write(",\n".join([json.dumps(line) for line in output_list]))
        f.write("]")

    # df.to_hdf(save_path, "ds")
    print(f"dataset saved to {filtered_save_path}")

    print


if __name__ == "__main__":
    main()

# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
import torch
from datasets import Dataset
from oracles import *  
from primary_models import get_m1
from secondary_model import (
    Repeater_Secondary_Model,
    OpenAI_Secondary_Model,
    Gt_Secondary_Model,
    Alpaca_Secondary_Model,
)
from utils import set_random_seed
from datetime import datetime
import numpy as np

from masking import (
    # adversarial_dataset,
    randsentence_dataset,
    randdist_dataset,
)
from pathlib import PurePath
import pandas as pd
import os
from preprocess import get_preprocessed_ds
from tqdm import tqdm

np.random.seed(42)


@click.command()
@click.option(
    "--split", default="validation", help="HotpotQA split {train, validation}"
)
@click.option("--m1_path", help="path to primary model")
@click.option("--m1_arch", help="primary model architecture")
@click.option(
    "--m2_arch", help="secondary model architecture {t5, gpt-3.5-turbo, gpt-4, alpaca, alexpaca, gt}"
)
@click.option("--alexpaca_path", help="path to trained alexpaca model", default=None)
@click.option(
    "--template_id",
    help="Which prompt template to use for the secondary model. {p1, p2, p3, p4, p5, p6}",
)
@click.option("--oracle_arch", default="t5", help="oracle architecture {t5, bloom}")
@click.option(
    "--oracle_size",
    help="oracle size, t5: {small, base, large, xl, xxl}",
)
@click.option("--pm_eval_batch_size", help="batch size for eval", default=1, type=int)
@click.option(
    "--oracle_eval_batch_size", help="batch size for eval", default=1, type=int
)
# @click.option(
#     "--max_adversarial_examples",
#     default=1,
#     help="create at most this many adversarial examples per example",
# )
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
    "--oai_cache_path",
    default=None,
    help="Path to save/load cached chatGPT responses.",
)
@click.option("--results_filename", help="path to save results")
@click.option("--save_dir", help="directory to save results to", default="results/")
@click.option(
    "--gt_subset", flag_value=True, help="filter in only gt examples for m2 comparisons"
)
def main(
    split,
    m1_path,
    m1_arch,
    m2_arch,
    alexpaca_path,
    template_id,
    oracle_arch,
    oracle_size,
    pm_eval_batch_size,
    oracle_eval_batch_size,
    max_adversarial_examples,
    downsample_pt_size,
    ds_shift,
    oai_cache_path,
    gt_subset,
    results_filename,
    save_dir,
):
    assert alexpaca_path and m2_arch == "alexpaca", "alexpaca path required iff m2_arch is alexpaca"
    masking_scheme = "randsentence"
    set_random_seed(0)
    if max_adversarial_examples is None:
        max_adversarial_examples = float("inf")
        print(
            "warning: failing to limit the number of adversarial examples may take a long time"
        )
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
    m1 = get_m1(m1_path, m1_arch, pm_eval_batch_size)
    # Receive and prepare the primary task
    metrics = {}

    print("preprocessing...")
    # ds = get_preprocessed_ds("validation", downsample_pt_size)
    assert split in ["train", "validation"]
    assert not (
        split == "train" and gt_subset
    ), "gt subset only works for validation since there are no ground truth examples in the training set"
    ds = get_preprocessed_ds(split)

    # filter out ids that don't appear in the gt dataset for speedup
    if m2_arch == "gt" or gt_subset:
        gt_df = pd.read_excel("gt_data/gt_dataset_v3_400_of_600.xlsx")  # excel
        # drop any gt without an m2
        gt_df = gt_df.dropna(subset=["q2_gt"])
        gt_df["masked_sentence"] = gt_df["masked_sentence"].apply(lambda x: x.strip())
        gt_ids = [x.split("_")[0] for x in gt_df["id"].tolist()]
        len_before = len(ds)
        ds = ds.filter(lambda x: x["id"].split("_")[0] in gt_ids)
        print(f"reduce to {len(ds)} / {len_before} examples")

    # downsample if a downsampling size is provided
    if str(downsample_pt_size) != "None":
        ds = ds.select(range(ds_shift, ds_shift + int(downsample_pt_size)))

    original_raw_dataset_len = len(ds)

    # first pass
    print("m1 first pass...")
    ds, metrics["supporting"] = m1.evaluate(
        masking_scheme="supporting", ds=ds, a2_col=None
    )
    # select and mask examples where the primary
    if masking_scheme == "bfsentence":
        raise NotImplementedError
    elif masking_scheme == "randsentence":
        do_gt = m2_arch == "gt" or gt_subset
        ds = randsentence_dataset(ds, m1, do_gt)
    elif masking_scheme == "randdistsentence":
        raise NotImplementedError
        ds = randdist_dataset(
            ds, m1, max_adversarial_examples
        )  # set drop thresh to -1 so no filtering happens

    # select only ground truth examples if we are doing analysis using the ground truth model
    if m2_arch == "gt" or gt_subset:
        # gt_df = pd.read_csv("q2_gt_dataset.csv")
        # gt_qs = set(gt_df["prepped_bfdelsentence_None"].tolist())
        gt_masked_sentences = set(gt_df["masked_sentence"].tolist())

        # gt_df["masked_sentence"][~gt_df["masked_sentence"].isin(set(ds["masked_sentence"]))] # 36 issues
        # df = ds.to_pandas()
        df = ds.to_pandas()
        before_len = len(ds)
        # filter based on the question cuz i forgot to include the full id in the labeling doc
        # and it wouldn't be ideal anyway cuz of suffix inconsistency
        # ds = ds.filter(lambda example: example["prepped_masked_None"] in gt_qs)
        ds = ds.filter(lambda x: x["masked_sentence"] in gt_masked_sentences)
        # select a random set of questions with one distractor added that match the annotated examples' masked sentences
        # df = ds.to_pandas()
        # pd.concat(list(x[1].head(1) for x in df.groupby("masked_sentence")))
        # dist_ds = retroactively_add_distractors(ds)
        print(len(set(ds["masked_sentence"])))
        # print(len(set(ds["distractor_sentence"])))
        print(len(set(ds["id"])))
        # get only the first example of each masked sentence
        used = set()
        relevant_examples = []
        for i, x in tqdm(enumerate(ds)):
            if x["masked_sentence"] not in used:
                relevant_examples.append(x)
                used.add(x["masked_sentence"])
        ds = Dataset.from_pandas(pd.DataFrame(data=relevant_examples))
        print
    # for gt dataset gen
    # tmp_df = ds.to_pandas()
    # tmp_df = tmp_df[[
    #     "id", "q1", "a1", "fc_masked", "masked_sentence", "masked_sentence_title"
    # ]].head(600)
    # tmp_df.to_csv("gt_dataset_source_v2_600.csv", index=False)
    # tmp_df.to_excel("gt_dataset_source_v2_600.xlsx", index=False)
    # Create the secondary model
    if m2_arch == "repeater":
        m2 = Repeater_Secondary_Model()
    elif m2_arch in ["gpt-3.5-turbo", "gpt-4"]:
        m2 = OpenAI_Secondary_Model(oai_cache_path, m2_arch, template_id)
    elif m2_arch == "gt":
        m2 = Gt_Secondary_Model(gt_df)
    elif m2_arch == "alpaca":
        m2 = Alpaca_Secondary_Model(
            "alpaca",
            ".model_cache/alpaca/tuned",
            prompt_id=template_id,
        )
    elif m2_arch == "alexpaca":
        m2 = Alpaca_Secondary_Model(
            "alpaca",
            alexpaca_path,
            # tokenizer_path=".model_cache/alpaca/tuned", # use the original alpaca tokenizer
            prompt_id="p1", # always use p1 since thats what it was trained on
        )
    else:
        raise NotImplementedError(f"m2_arch {m2_arch} not implemented")

    # Apply the secondary model
    print("m2...")
    ds = m2.process(
        ds,
        q1_col="q1",
        masking_scheme="masked",
    )
    # Save memory by moving m1 to CPU
    if type(m1.model) == T5ForConditionalGeneration:
        m1.model.cpu()
    # del m1
    torch.cuda.empty_cache()  # free up memory
    # Create the oracle
    if oracle_arch == "t5":
        oracle = T5_Bool_Oracle(
            model_size=oracle_size, batch_size=oracle_eval_batch_size
        )
    elif oracle_arch == "gpt-3.5-turbo":
        oracle = OpenAI_Oracle("gpt-3.5-turbo")
    elif oracle_arch == "gpt-4":
        oracle = OpenAI_Oracle("gpt-4")
    else:
        raise NotImplementedError

    # Answer questions with the oracle
    print("oracle...")
    ds = oracle.process(ds, q2_masking_scheme="masked")
    if type(oracle.model) == T5ForConditionalGeneration:
        oracle.model.cpu()
    torch.cuda.empty_cache()  # free up memory

    # Bring back the primary model
    m1 = get_m1(m1_path, m1_arch, pm_eval_batch_size)
    # Evaluate the primary model on the masked examples
    print("m1 second pass...")
    ds, metrics["masked"] = m1.evaluate(masking_scheme="masked", ds=ds, a2_col=None)

    # Evaluate the primary model on the answered examples
    print("m1 third pass...")
    ds, metrics["answered"] = m1.evaluate(masking_scheme="masked", ds=ds, a2_col="a2")

    # Analysis
    df = pd.DataFrame(ds)
    print(f"runtime: {datetime.now()-start}")

    # make the dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(
        save_dir,
        f"analysis_dataset_{'full' if str(downsample_pt_size) == 'None' else downsample_pt_size}_{masking_scheme}_{m1_arch}_{m2_arch}_{oracle_arch}_{oracle_size}_{template_id}.hd5",
    )
    desrcribe_path = PurePath(save_path + "_" + str(datetime.now())).with_suffix(".csv")
    describe_df = (
        df[
            [
                "m1_supporting_None_f1",
                "m1_masked_None_f1",
                "m1_masked_a2_f1",
                "m1_supporting_None_em",
                "m1_masked_None_em",
                "m1_masked_a2_em",
                "a2_is_correct_masked",
            ]
        ]
        .astype(float)
        .describe()
    )
    describe_df.to_csv(desrcribe_path)

    df.to_hdf(save_path, "ds")
    print(f"dataset saved to {save_path}")

    print


if __name__ == "__main__":
    main()

# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
import torch
from primary_models import get_m1
from oracles import *  # T5_Bool_Oracle
from secondary_model import Llama3_Secondary_Model
from main import main as analyze
from utils import set_random_seed
from datetime import datetime
import numpy as np
from datasets import Dataset
from masking import randsentence_dataset
from pathlib import PurePath
import pandas as pd
import os
from preprocess import get_preprocessed_ds
from tqdm import tqdm
import re
import json


np.random.seed(0)


@click.command()
@click.option(
    "--split", default="train", help="HotpotQA split {train, validation}"
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
@click.option("--save_dir", help="directory to save results to", default="data/zach")
@click.option(
    "--gt_subset", flag_value=True, help="filter in only gt examples for m2 comparisons"
)
@click.option(
    "--rounds",
    default=20,
    help="number of rounds of active filtering",
)
@click.option(
    "--initial_seed",
    default=0,
    help="starting seed for rejection sampling",
)
@click.option(
    "--do_temp_scaling",
    default=True,
    help="whether to scale temperature for rejection sampling",
)
def main(
    split,
    m2_arch,
    downsample_pt_size,
    ds_shift,
    gt_subset,
    save_dir,
    rounds,
    initial_seed,
    do_temp_scaling
):
    set_random_seed(initial_seed)

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

    do_gt = m2_arch == "gt" or gt_subset
    m1 = None
    ds = randsentence_dataset(ds, m1, do_gt)

    masked_sentences = ds["masked_sentence"]
    # downsample and shift
    print("loading llama3 model...")

    m1 = get_m1("t5-base", batch_size=64)
    # get baseline m1 performance
    print("m1 first pass...")
    ds, _ = m1.evaluate(masking_scheme="masked", ds=ds, a2_col=None)
    del m1

    df = pd.DataFrame(ds).set_index("id")
    df["idx"] = range(len(df))
    df["filtered_q2"] = [None] * len(df)
    df["round"] = [None] * len(df)
    ds = Dataset.from_pandas(df)
    running_q2s = []
    for i in range(rounds):
        needs_q2_idx = df[df["filtered_q2"].isnull()]["idx"]
        active_ds = ds.select(needs_q2_idx)
        llama3 = Llama3_Secondary_Model(
            # "NO_MODEL_PATH",
            prompt_id="p3",
            eval_batch_size=1,
            temperature=i / rounds + 0.01 if do_temp_scaling else 0.01,
            do_sample=True,
        )
        active_ds = llama3.process(active_ds, "q1")
        running_q2s.append(pd.DataFrame(active_ds).set_index("id")["q2"])
        del llama3
        oracle = T5_Bool_Oracle(
            batch_size=64,
        )
        active_ds = oracle.process(active_ds)
        del oracle
        m1 = get_m1("t5-base", batch_size=64)
        active_ds, _ = m1.evaluate(masking_scheme="masked", ds=active_ds, a2_col="a2")
        del m1
        active_df = pd.DataFrame(active_ds)
        improved_id = active_df[
            # E1
            # ((active_df["m1_masked_a2_f1"] - active_df["m1_masked_None_f1"] > 0)
            # | (active_df["m1_masked_a2_f1"] == 1))

            # E2
            # ((active_df["m1_masked_a2_f1"] - active_df["m1_masked_None_f1"] > 0)
            # & active_df["m1_masked_None_f1"] == 0

            # E4
            active_df["a2_is_correct"]
            
            # E5
            # ((active_df["m1_masked_a2_f1"] - active_df["m1_masked_None_f1"] > 0)
            # | (active_df["m1_masked_a2_f1"] > 0.5))
            # | active_df["a2_is_correct"]
        ]["id"]
        print(f"it: {i}\timproved {len(improved_id)}/{len(ds)}")
        df.loc[improved_id, "filtered_q2"] = active_df.set_index("id").loc[
            improved_id, "q2"
        ]
        df.loc[improved_id, "round"] = i
        # increment the seed
        set_random_seed(i + 1)

    # make the dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save_path = os.path.join(
    #     save_dir,
    #     f"jeopardy_{'full' if str(downsample_pt_size) == 'None' else downsample_pt_size}_{split}.jsonl",
    # )

    filtered_save_path = os.path.join(
        save_dir,
        f"jeopardy_{'full' if str(downsample_pt_size) == 'None' else downsample_pt_size}_{split}_zach_filtered_tatsu_seed-{initial_seed}_E12.jsonl",
    )
    # df.to_json(filtered_save_path, orient="records", lines=True)

    # df["instruction"] = df.apply(
    #     lambda x: fit_template(x["q1"], x["fc_masked"]), axis=1
    # )
    df = df[df["filtered_q2"].notnull()]
    df["output"] = df["filtered_q2"]
    df["input"] = ""

    # write instructions, output, and input to a jsonl file as a list of dicts
    output_list = df[["q1", "fc_masked", "input", "output", "round"]].to_dict(orient="records")
    with open(filtered_save_path, "w") as f:
        f.write("[")
        f.write(",\n".join([json.dumps(line) for line in output_list]))
        f.write("]")

    # df.to_hdf(save_path, "ds")
    print(f"dataset saved to {filtered_save_path}")

    print


if __name__ == "__main__":
    main()

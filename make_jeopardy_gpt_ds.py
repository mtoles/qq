# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
import torch
from datasets import load_from_disk, Dataset
from secondary_model import (
    OpenAI_Secondary_Model,
    OpenAI_Jeopardy_Secondary_Model,
)
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
from tqdm import tqdm
import json

np.random.seed(42)


@click.command()
@click.option(
    "--split", default="validation", help="HotpotQA split {train, validation}"
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
    "--oai_cache_path",
    default=None,
    help="Path to save/load cached chatGPT responses.",
)
@click.option(
    "--save_dir", help="directory to save results to", default="data/jeopardy"
)
@click.option(
    "--gt_subset", flag_value=True, help="filter in only gt examples for m2 comparisons"
)
def main(
    split,
    downsample_pt_size,
    ds_shift,
    oai_cache_path,
    gt_subset,
    save_dir,
):
    set_random_seed(0)
    if ds_shift:
        assert (
            downsample_pt_size is not None
        ), "There is no reason to shift the dataset without downsampling"
    now = datetime.now().strftime("Y%m%d-%H%M%S")

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

    print("masking...")
    ds = randsentence_dataset(ds, None, False)

    masked_sentences = ds["masked_sentence"]
    # downsample and shift
    print("loading language model...")
    model = OpenAI_Secondary_Model(
        ".cache/shelved_cache",
        "gpt-4",
    )
    # model = OpenAI_Jeopardy_Secondary_Model(
    #     None,
    #     "gpt-4-turbo",
    # )
    jeopardy_qs = []
    print("generating jeopardy questions...")
    for i in tqdm(range(0, len(ds))):
        batch = ds[i]
        masked_sentences = batch["masked_sentence"]
        batch_output = model.forward(batch, "q1", "fc_masked")
        # batch_output = model.forward(batch, "q1", "fc_masked", "masked_sentence")
        jeopardy_qs.append(batch_output)

    ds = ds.add_column("output", jeopardy_qs)

    # keep only necessary cols

    # # Analysis
    df = pd.DataFrame(ds)
    # print(f"runtime: {datetime.now()-start}")

    # make the dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(
        save_dir,
        f"gpt_{'full' if str(downsample_pt_size) == 'None' else downsample_pt_size}_{split}.jsonl",
    )

    output_list = df[["q1", "fc_masked", "output"]].to_dict(orient="records")
    with open(save_path, "w") as f:
        f.write("[")
        f.write(",\n".join([json.dumps(line) for line in output_list]))
        f.write("]")

    # df.to_hdf(save_path, "ds")
    print(f"dataset saved to {save_path}")

    print


if __name__ == "__main__":
    main()

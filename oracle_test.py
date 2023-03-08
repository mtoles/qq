# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
from datasets import load_from_disk, Dataset
from oracles import Dummy_Oracle, T5_Oracle
from secondary_model import (
    Repeater_Secondary_Model,
    OpenAI_Secondary_Model,
    Gt_Secondary_Model,
)
from primary_models import BigBird_PM, T5_PM
from dataset_utils import drop_unanswerable
from datetime import datetime

import pandas as pd
import numpy as np


@click.command()
@click.option(
    "--raw_dataset_path", help="path to the raw dataset output by preprocess.py"
)
@click.option("--gt_dataset_csv_path", help="path to csv containing q2_gt column")
@click.option("--oracle_arch", help="oracle architecture")
@click.option("--eval_batch_size", default=2, help="batch size for eval")
@click.option(
    "--downsample_pt_size",
    default=None,
    help="use at most this many examples in validation",
)
@click.option("--results_filename", help="path to save results")
# @click.option("--results_filename", help="path to save results")
def main(
    raw_dataset_path,
    gt_dataset_csv_path,
    oracle_arch,
    eval_batch_size,
    downsample_pt_size,
    results_filename,
):
    gt_csv = pd.read_csv("q2_gt_dataset.csv")
    raw_dataset = pd.DataFrame(load_from_disk(raw_dataset_path))
    if str(downsample_pt_size) != "None":
        raw_dataset = raw_dataset.select(range(int(downsample_pt_size)))
    df = gt_csv.merge(raw_dataset, on="id").dropna()
    
    # new_df["context_distractor"] = compare_400_csv["context_distractor"]

    now = datetime.now().strftime("Y%m%d-%H%M%S")
    if results_filename is None:
        results_filename = f"{oracle_arch}-{downsample_pt_size}-{now}"

    with open(f"inf_logs/{results_filename}.txt", "a") as f:

        assert oracle_arch.startswith("t5") or oracle_arch == "dummy"

        # Receive and prepare the primary task
        df = pd.read_csv(gt_dataset_csv_path)

        # Drop examples without gt q2
        df = df.dropna(subset=["q2_gt"])
        if str(downsample_pt_size) != "None":
            df = df.iloc[:downsample_pt_size]

        ds = Dataset.from_pandas(df)

        # Create the oracle
        oracle = T5_Oracle(model_name=oracle_arch)
        # Answer questions with the oracle
        ds = oracle.process(ds, q2_masking_scheme="gt")
        print


if __name__ == "__main__":
    main()

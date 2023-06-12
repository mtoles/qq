# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
from datasets import load_from_disk, Dataset
from oracles import *
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
from transformers import pipeline


# def get_oracle_by_arch(oracle_arch, oracle_size, eval_batch_size):
# arch_map = {
#     "t5-small-gen": "t5-small",
#     "t5-xxl-gen": "t5-xxl",
#     "t5-small-bool": "t5-small",
#     "t5-xxl-bool": "t5-xxl",
#     "word-overlap": "word-overlap",
# }
# class_map = {
#     "t5-small-gen": T5_Gen_Oracle,
#     "t5-xxl-gen": T5_Gen_Oracle,
#     "t5-small-bool": T5_Bool_Oracle,
#     "t5-xxl-bool": T5_Bool_Oracle,
#     "word-overlap": Word_Overlap_Oracle,
# }
# oracle_name = arch_map[oracle_arch]
# if oracle_name.startswith("t5"):
#     oracle = class_map[oracle_arch](oracle_name, eval_batch_size)
# elif oracle_name == "word-overlap":
#     oracle = class_map[oracle_arch](eval_batch_size)
# return oracle


@click.command()
@click.option(
    "--raw_dataset_path", help="path to the raw dataset output by preprocess.py"
)
@click.option("--gt_dataset_csv_path", help="path to csv containing q2_gt column")
@click.option("--oracle_arch", help="oracle architecture, e.g. t5")
@click.option("--oracle_size", help="oracle name, e.g. xxl")
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
    oracle_size,
    eval_batch_size,
    downsample_pt_size,
    results_filename,
):







    gt_csv = pd.read_csv("q2_gt_dataset.csv")
    raw_dataset = pd.DataFrame(load_from_disk(raw_dataset_path))
    if str(downsample_pt_size) != "None":
        raw_dataset = raw_dataset.select(range(int(downsample_pt_size)))
    df = raw_dataset.merge(gt_csv, on="id").dropna()

    # new_df["context_distractor"] = compare_400_csv["context_distractor"]

    now = datetime.now().strftime("Y%m%d-%H%M%S")
    # if results_filename is None:
    #     results_filename = f"{oracle_arch}-{downsample_pt_size}-{now}"

    # with open(f"inf_logs/{results_filename}.txt", "a") as f:

    # assert oracle_arch.startswith("t5") or oracle_arch == "dummy"

    # Drop examples without gt q2
    df = df.rename(columns={"gt_q2": "q2_gt"})
    df = df.dropna(subset=["q2_gt"])
    if str(downsample_pt_size) != "None":
        df = df.iloc[:downsample_pt_size]

    # figure out the 'masked_sentence_title' col since it was not recorded during annotation
    def get_masked_sentence_index(row):
        article_count = 0
        for li in row["context_None"]["sentences"]:
            for sent in li:
                if sent == row["masked_sentence"]:
                    return article_count
            article_count += 1
        raise Exception("masked sentence not found")

    def get_masked_sentence_title(row):
        masked_sentence_index = get_masked_sentence_index(row)
        title = row["context_None"]["title"][masked_sentence_index]
        return title

    df["masked_sentence_title"] = df.apply(get_masked_sentence_title, axis=1)

    ds = Dataset.from_pandas(df)

    # Create the oracle
    oracle_name = f"{oracle_arch}-{oracle_size}"
    # Oracle = {
    #     "t5": T5_Bool_Oracle,
    #     "bloom": Bloom_Bool_Oracle,
    #     "dolly": Dolly_Bool_Oracle,
    # }[oracle_arch]

    # spot testing
    generate_text = pipeline(model=oracle_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device=0)
    context = "George Washington (February 22, 1732[b] - December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. When was George Washington president?"
    res = generate_text(context)
    print(res[0]["generated_text"])
    
    res = generate_text("A: How many species of kangaroo are there? \n\nB: elephants are grey. \n\n Prompt: Is B a good answer for A, yes or no?")[0]["generated_text"]
    print(res)
    oracle = Oracle(model_size=oracle_size, batch_size=eval_batch_size)

 


    # Answer questions with the oracle    
    ds = oracle.process(ds, q2_masking_scheme="gt")

    percent_correct = sum(ds["a2_is_correct_gt"]) / len(ds)
    print(f"Oracle accuracy: %.3f%%" % (percent_correct * 100))
    print


if __name__ == "__main__":
    main()

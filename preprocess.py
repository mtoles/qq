"""Preprocess the dataset storing formatted examples with masking in text form"""

import os
import pandas as pd

import numpy as np
from tqdm import tqdm

import click
from datasets import load_dataset
import datasets
from typing import List, Optional, Tuple
from collections import defaultdict, Counter

from masking import mask_random_sentence, split_distractor
from utils import (
    make_cache_file_name,
    get_downsample_dataset_size_str,
    CATEGORY_MAPPING,
)


DOC_STRIDE = 2048
MAX_LENGTH = 4096
SEED = 42
PROCESS_TRAIN = os.environ.pop("PROCESS_TRAIN", "false")


def flatten_context(example, masking_scheme):
    masking_str = f"context_{masking_scheme}"
    output = ""
    titles = example[masking_str]["title"]  # list of str
    sentences = example[masking_str]["sentences"]  # list of list of str
    paragraphs = [" ".join(s) for s in sentences]
    contexts = [f"{t}: {p}" for t, p in zip(titles, paragraphs)]
    context = "\n\n".join(contexts)
    context = " [SEP] ".join([example["question"], context])
    return {f"fc_{masking_scheme}": context}


@click.command()
@click.option("--split", type=str, help="{train | validation | both}")
@click.option("--dataset", type=str, help="{natural_questions | hotpot}")
@click.option("--masking_schemes", type=str, multiple=True, default=None)
@click.option("--distract_or_focus", type=str, help="{distract | focus}")
@click.option("--downsample_data_size", type=str, default=None)
@click.option("--cache_dir", type=str, help="Path to cache directory")
@click.option("--load_from_cache", type=bool, default=True)
def main(
    split,
    dataset,
    masking_schemes,
    downsample_data_size,
    distract_or_focus,
    cache_dir,
    load_from_cache,
):
    # Unit Tests

    assert split in ["train", "validation"], "Invalid split"
    assert dataset in ["hotpot"], "Invalid dataset"
    assert "None" not in masking_schemes, "`None` masking will be included by default."
    masking_dict = {"randomsentence": mask_random_sentence}
    for masking_scheme in masking_schemes:
        assert (
            masking_scheme in masking_dict.keys()
        ), f"Invalid masking scheme {masking_scheme}"
    assert distract_or_focus in ["distract", "focus"], "Invalid distract_or_focus"

    # Prep data dir
    if not os.path.exists("data/preprocess"):
        os.makedirs("data/preprocess")

    # Load the Dataset

    raw_dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        cache_dir=cache_dir,
        split=f"{split}{get_downsample_dataset_size_str(downsample_data_size)}",
    )
    new_ds = raw_dataset.rename_column("context", "context_None")

    cache_file_name = os.path.join(
        "data",
        "preprocess",
        make_cache_file_name(
            split, dataset, downsample_data_size, masking_schemes, distract_or_focus
        ),
    )

    # Drop Distractor Content
    print("Dropping distractor sentences...")
    if distract_or_focus == "focus":
        new_ds = new_ds.add_column("context_distractor", [{} for _ in range(len(new_ds))])
        new_ds = new_ds.map(
            split_distractor,
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache,
        )

    # Apply Each Masking Scheme

    for masking_scheme in masking_schemes:
        masking_str = f"context_{masking_scheme}"

        masking_fn = masking_dict[masking_scheme]
        print(f"Applying masking scheme {masking_scheme}...")
        new_ds = new_ds.add_column(name="masked_sentence", column=["" for _ in range(len(new_ds))])
        new_ds = new_ds.add_column(name="context_randomsentence", column=[{} for _ in range(len(new_ds))])
        # masked_col = new_ds.map(
        #     masking_fn,
        #     cache_file_name=cache_file_name,
        #     load_from_cache_file=load_from_cache,
        # )["masked_col"]
        # new_ds = new_ds.add_column(name=masking_str, column=masked_col)
        new_ds = new_ds.map(masking_fn, cache_file_name=cache_file_name, load_from_cache_file=load_from_cache)

    # Flatten Each Context

    for masking_scheme in list(masking_schemes) + ["None"]:
        masking_str = f"context_{masking_scheme}"
        print(f"Flattening context {masking_scheme}...")
        flat_col = new_ds.map(
            lambda x: flatten_context(x, masking_scheme),
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache,
        )[
            f"fc_{masking_scheme}"
        ]  # fc == flattened context
        new_ds = new_ds.add_column(name=f"fc_{masking_scheme}", column=flat_col)
        new_ds = new_ds.remove_columns([f"context_{masking_scheme}"])

    # Normalize Whitespace
    print("Normalizing whitespace...")
    new_ds = new_ds.map(
        lambda x: {f"fc_{masking_scheme}": " ".join(x[f"fc_{masking_scheme}"].split())},
        cache_file_name=cache_file_name,
        load_from_cache_file=load_from_cache,
    )

    save_path = os.path.join(
        "data",
        "preprocess",
        make_cache_file_name(
            dataset, split, downsample_data_size, masking_schemes, distract_or_focus
        ),
    )
    new_ds.save_to_disk(save_path)


if __name__ == "__main__":
    main()

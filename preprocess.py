"""Preprocess the dataset storing formatted examples with masking in text form"""

import os
import click
from datasets import load_dataset

from masking import mask_None, split_distractor, add_flat_contexts
from utils import (
    make_cache_file_name,
    get_downsample_dataset_size_str,
    CATEGORY_MAPPING,
)


# @click.command()
# @click.option("--split", type=str, help="{train | validation | both}")
# # @click.option("--distract_or_focus", type=str, help="{distract | focus}")
# @click.option("--downsample_data_size", type=str, default=None)
# # @click.option("--cache_dir", type=str, help="Path to cache directory")
# # @click.option("--load_from_cache", type=bool, default=True)
def get_preprocessed_ds(
    split,
    # downsample_data_size,
    # distract_or_focus,
    # cache_dir,
    # load_from_cache,
):
    # Unit Tests
    assert split in ["train", "validation"], "Invalid split"
    # assert distract_or_focus in ["distract", "focus"], "Invalid distract_or_focus"

    # Prep data dir
    if not os.path.exists("data/preprocess"):
        os.makedirs("data/preprocess")

    # Load the Dataset

    raw_dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        cache_dir=".cache",
        # split=f"{split}{get_downsample_dataset_size_str(downsample_data_size)}",
        split=f"{split}",
    )
    new_ds = raw_dataset.rename_column("context", "context_None")

    # cache_file_name = os.path.join(
    #     "data",
    #     "preprocess",
    #     make_cache_file_name(
    #         split, dataset, downsample_data_size, masking_schemes, distract_or_focus
    #     ),
    # )

    # Drop Distractor Content
    print("Splitting out distractor sentences...")
    # assert distract_or_focus == "focus", "distract not implemented yet"
    new_ds = new_ds.add_column("context_distractor", [{} for _ in range(len(new_ds))])
    new_ds = new_ds.add_column("context_supporting", [{} for _ in range(len(new_ds))])
    new_ds = new_ds.map(
        split_distractor,
        load_from_cache_file=False,
    )

    # Flatten Each Context

    new_ds = add_flat_contexts(
        new_ds,
        ["None", "supporting", "distractor"],
    )

    # filter out examples with no distractors
    before = len(new_ds)
    new_ds = new_ds.filter(
        lambda ex: max([len(x) for x in ex["context_distractor"]["sentences"]]) > 0,
        load_from_cache_file=False,
    )
    print(f"Filtered out {before - len(new_ds)} examples with no distractors")

    # save_path = os.path.join(
    #     "data",
    #     "preprocess",
    #     make_cache_file_name(
    #         dataset, split, downsample_data_size, masking_schemes, distract_or_focus
    #     ),
    # )

    # new_ds.save_to_disk(save_path)
    return new_ds


if __name__ == "__main__":
    get_preprocessed_ds()

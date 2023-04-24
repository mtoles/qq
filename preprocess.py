"""Preprocess the dataset storing formatted examples with masking in text form"""

import os
import click
from datasets import load_dataset

from masking import mask_random_sentence, mask_None, split_distractor, add_flat_contexts
from utils import (
    make_cache_file_name,
    get_downsample_dataset_size_str,
    CATEGORY_MAPPING,
)


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
    masking_schemes = list(masking_schemes)
    assert split in ["train", "validation"], "Invalid split"
    assert dataset in ["hotpot"], "Invalid dataset"
    assert "None" not in masking_schemes, "`None` masking will be included by default."
    # masking_schemes = list(masking_schemes) + ["None"]
    masking_dict = {"randomsentence": mask_random_sentence, "None": mask_None}
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
    print("Splitting out distractor sentences...")
    assert distract_or_focus == "focus", "distract not implemented yet"
    new_ds = new_ds.add_column("context_distractor", [{} for _ in range(len(new_ds))])
    new_ds = new_ds.add_column("context_supporting", [{} for _ in range(len(new_ds))])
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

        # empty columns to be filled in by the masking function
        if "masked_sentence" not in new_ds.column_names:
            new_ds = new_ds.add_column(
                name="masked_sentence", column=["" for _ in range(len(new_ds))]
            )
        if "context_randomsentence" not in new_ds.column_names:
            new_ds = new_ds.add_column(
                name="context_randomsentence", column=[{} for _ in range(len(new_ds))]
            )
        # masked_col = new_ds.map(
        #     masking_fn,
        #     cache_file_name=cache_file_name,
        #     load_from_cache_file=load_from_cache,
        # )["masked_col"]
        # new_ds = new_ds.add_column(name=masking_str, column=masked_col)
        new_ds = new_ds.map(
            masking_fn,
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache,
        )

    # Flatten Each Context

    new_ds = add_flat_contexts(
        new_ds,
        masking_schemes + ["None", "supporting", "distractor"],
        cache_file_name,
        load_from_cache,
    )

    # filter out examples with no distractors
    before = len(new_ds)
    new_ds = new_ds.filter(
        lambda ex: max([len(x) for x in ex["context_distractor"]["sentences"]]) > 0
    )
    print(f"Filtered out {before - len(new_ds)} examples with no distractors")

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

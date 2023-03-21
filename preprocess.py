"""
Preprocess the dataset storing formatted examples with masking in text form

Masking types:
    - None: No masking
    - randomsentence: Mask a random sentence. Sentence is saved in the `masked_sentence` column.
    - bfsentence: Mask every sentence. Note that this increased the length of the dataset by a factor of (num supporting sentences). Sentence is saved in the `masked_sentence` column.

"""

import os
import click
from datasets import load_dataset, concatenate_datasets

from masking import mask_random_sentence, mask_bf_sentence, mask_None, split_distractor
from utils import (
    make_cache_file_name,
    get_downsample_dataset_size_str,
    CATEGORY_MAPPING,
)
from tqdm import tqdm


def flatten_context(example, masking_scheme):
    masking_str = f"context_{masking_scheme}"
    titles = example["context_None"]["title"]  # list of str
    sentences = example[masking_str]["sentences"]  # list of list of str
    paragraphs = [" ".join(s) for s in sentences]
    contexts = [f"{t}: {p}" for t, p in zip(titles, paragraphs) if p]
    context = "\n\n".join(contexts)
    return {f"fc_{masking_scheme}": context}


@click.command()
@click.option("--split", type=str, help="{train | validation | both}")
@click.option("--dataset", type=str, help="{ hotpot }")
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
    assert distract_or_focus in ["distract", "focus"], "Invalid distract_or_focus"
    if "bfsentence" in masking_schemes:
        assert (
            len(masking_schemes) == 1
        ), "Cannot use other masking schemes with bruteforce since bruteforce changes the length of the dataset"
        do_bf = True
    else:
        do_bf = False
    masking_dict = {
        "randomsentence": mask_random_sentence,
        "None": mask_None,
        "bfsentence": mask_bf_sentence,
    }
    for masking_scheme in masking_schemes:
        assert (
            masking_scheme in masking_dict.keys()
        ), f"Invalid masking scheme {masking_scheme}"

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
    # drop new_ds[i] if new_ds[i]["context_distractor"]["sentences"] consists only of empty lists, e.g. [[], []]
    before_len = len(new_ds)
    new_ds = new_ds.filter(
        lambda example: any(
            [len(s) > 0 for s in example["context_distractor"]["sentences"]]
        )
    )
    after_len = len(new_ds)
    print(
        f"Filtered out {before_len - after_len} of {before_len} examples with no distractor sentences."
    )

    # empty columns to be filled in by the masking function
    if "masked_sentence" not in new_ds.column_names:
        new_ds = new_ds.add_column(
            name="masked_sentence", column=["" for _ in range(len(new_ds))]
        )
    if (
        "context_randomsentence" not in new_ds.column_names
        and "randomsentence" in masking_schemes
    ):
        new_ds = new_ds.add_column(
            name="context_randomsentence", column=[{} for _ in range(len(new_ds))]
        )
    if (
        "context_bfsentence" not in new_ds.column_names
        and "bfsentence" in masking_schemes
    ):
        new_ds = new_ds.add_column(
            name="context_bfsentence", column=[{} for _ in range(len(new_ds))]
        )

    # Apply Each Masking Scheme
    if do_bf:
        # TODO: speed up
        print("Applying bruteforce masking...")
        bf_mini_datasets = [
            masking_dict[masking_scheme](example) for example in tqdm(new_ds)
        ]

        new_ds = concatenate_datasets(bf_mini_datasets)
    else:
        for masking_scheme in masking_schemes:
            masking_str = f"context_{masking_scheme}"

            masking_fn = masking_dict[masking_scheme]
            print(f"Applying masking scheme {masking_scheme}...")

            new_ds = new_ds.map(
                masking_fn,
                cache_file_name=cache_file_name,
                load_from_cache_file=load_from_cache,
            )

    # Flatten Each Context

    for masking_scheme in list(masking_schemes) + ["None", "supporting", "distractor"]:
        masking_str = f"context_{masking_scheme}"
        print(f"Flattening context {masking_scheme}...")
        flat_col = new_ds.map(
            lambda x: flatten_context(x, masking_scheme),
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache,
        )[
            f"fc_{masking_scheme}"
        ]  # fc == flattened context
        if f"fc_{masking_scheme}" not in new_ds.column_names:
            new_ds = new_ds.add_column(name=f"fc_{masking_scheme}", column=flat_col)
        # new_ds = new_ds.remove_columns([f"context_{masking_scheme}"])

    # Normalize Whitespace
    print("Normalizing whitespace...")
    for masking_scheme in list(masking_schemes) + ["None", "supporting", "distractor"]:
        # masking_str = f"context_{masking_scheme}"
        new_ds = new_ds.map(
            lambda x: {
                f"fc_{masking_scheme}": " ".join(x[f"fc_{masking_scheme}"].split())
            },
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache,
        )

    # Rename Columns
    new_ds = new_ds.rename_column("question", "q1")
    new_ds = new_ds.rename_column("answer", "a1")

    save_path = os.path.join(
        "data",
        "preprocess",
        make_cache_file_name(
            dataset, split, downsample_data_size, masking_schemes, distract_or_focus
        ),
    )
    print("saving dataset to", save_path)
    new_ds.save_to_disk(save_path)


if __name__ == "__main__":
    main()

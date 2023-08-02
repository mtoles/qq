from collections import defaultdict
from numpy import random
from copy import deepcopy
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

import pandas as pd
import numpy as np

from dataset_utils import combine_adversarial_ds
from datasets.utils.logging import disable_progress_bar, enable_progress_bar


# Maximum number of adversarial examples given each unique example id.
# Masked examples and distracted examples are counted separately.
col_name_map = {
    "m1_bfdelsentence_None_gen": "m1_masked_None_gen",
    "m1_bfdelsentence_None_f1": "m1_masked_None_f1",
    "m1_bfdelsentence_None_em": "m1_masked_None_em",
    "fc_bfdelsentence": "fc_masked",
}


def flatten_context(example, masking_scheme):
    masking_str = f"context_{masking_scheme}"
    titles = example["context_None"]["title"]  # list of str
    sentences = example[masking_str]["sentences"]  # list of list of str
    paragraphs = [" ".join(s) for s in sentences]
    contexts = [f"{t}: {p}" for t, p in zip(titles, paragraphs) if p]
    context = "\n\n".join(contexts)
    # context = " [SEP] ".join([example["question"], context])
    return {f"fc_{masking_scheme}": context}


def add_flat_contexts(
    new_ds, masking_schemes, cache_file_name=None, load_from_cache=False
):
    """Convert lists of sentences into single strings under the prefix "fc_". Used both here and in preprocess.py"""
    for masking_scheme in list(masking_schemes):
        masking_str = f"context_{masking_scheme}"
        # print(f"Flattening context {masking_scheme}...")
        disable_progress_bar()
        flat_col = new_ds.map(
            lambda x: flatten_context(x, masking_scheme),
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache,
        )[
            f"fc_{masking_scheme}"
        ]  # fc == flattened context
        enable_progress_bar()
        if f"fc_{masking_scheme}" not in new_ds.column_names:
            new_ds = new_ds.add_column(name=f"fc_{masking_scheme}", column=flat_col)
        # new_ds = new_ds.remove_columns([f"context_{masking_scheme}"])

    # Normalize Whitespace
    # print("Normalizing whitespace...")
    # TODO: the list of three should be removed and sent in the call
    for masking_scheme in list(masking_schemes):
        # masking_str = f"context_{masking_scheme}"
        disable_progress_bar()
        new_ds = new_ds.map(
            lambda x: {
                f"fc_{masking_scheme}": " ".join(x[f"fc_{masking_scheme}"].split())
            },
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache,
        )
        enable_progress_bar()
    # Rename Columns
    if "question" in new_ds.column_names:
        assert (
            "q1" not in new_ds.column_names
        ), "q1 already exists. You probably ran this twice."
        new_ds = new_ds.rename_column("question", "q1")
    if "answer" in new_ds.column_names:
        assert (
            "a1" not in new_ds.column_names
        ), "a1 already exists. You probably ran this twice."
        new_ds = new_ds.rename_column("answer", "a1")
    return new_ds


def adversarial_dataset(ds, m1, adversarial_drop_thresh, max_adversarial_examples):
    # if you use {} instead of {"sentences": []} you encounter a bug in datasets.Dataset.map() version 2.10.1
    ds = ds.add_column(
        "context_bfdelsentence", [{"sentences": []} for _ in range(len(ds))]
    )
    ds = ds.add_column(
        "context_bfaddsentence", [{"sentences": []} for _ in range(len(ds))]
    )

    # masking

    ds_got_right_supporting = ds.filter(
        lambda x: x["m1_supporting_None_f1"] > 0.0, load_from_cache_file=False
    )
    ds_bfdelsentence = bf_del_sentences(ds_got_right_supporting)
    # Don't run masking in adversarial mode so that we get a more
    # even distribution once we run it again in distractor mode.
    print("m1 masking...")
    ds_bfdelsentence, _metrics = m1.evaluate(
        masking_scheme="bfdelsentence", ds=ds_bfdelsentence, a2_col=None
    )

    disable_progress_bar()
    ds_got_worse_with_bfdelsentence = ds_bfdelsentence.filter(
        lambda x: x["m1_supporting_None_f1"] - x["m1_bfdelsentence_None_f1"]
        > adversarial_drop_thresh
    )

    # distracting

    ds_got_right_masked = ds_bfdelsentence.filter(
        lambda x: x["m1_bfdelsentence_None_f1"] > adversarial_drop_thresh,
        load_from_cache_file=False,
    )  # filtering not technically necessary but speeds things up
    mini_dss_bfaddsentence = bf_add_sentences(ds_got_right_masked)

    mini_dss = []

    # mini_dss_from_sup = []
    print("m1 distracting...")
    # for mini, li in [
    #     (mini_dss_bfaddsentence, mini_dss),
    #     (mini_dss_bfaddsentence_from_sup, mini_dss_from_sup),
    # ]:
    for mini_ds in tqdm(mini_dss_bfaddsentence):
        # ds_got_worse_with_bf_add_sentence, metrics["bfaddsentence"] = m1.evaluate(
        mini_ds_bf_add_sentence, _metrics = m1.evaluate(
            masking_scheme="bfaddsentence",
            ds=mini_ds,
            a2_col=None,
            max_adversarial_examples=max_adversarial_examples,
            threshold=adversarial_drop_thresh,
            display=False
        )
        mini_dss.append(mini_ds_bf_add_sentence)
    ds_got_worse_with_bf_add_sentence = concatenate_datasets(mini_dss)

    # evaluate the reference col
    ds_got_worse_with_bf_add_sentence, _metrics = m1.evaluate(
        masking_scheme="suppbfaddsentence",
        ds=ds_got_worse_with_bf_add_sentence,
        a2_col=None,
        max_adversarial_examples=None,
        threshold=None,
    )

    # create a reference copy of mini_dss_bfaddsentence that uses the the supporting context
    # instead so that we can filter properly later
    # mini_dss_bfaddsentence_from_sup =

    # ds_got_worse_with_bf_add_sentence_from_sup = concatenate_datasets(mini_dss_from_sup)
    # print(sum(ds_got_worse_with_bf_add_sentence["m1_bfaddsentence_None_f1"]))
    # print(sum(ds_got_worse_with_bf_add_sentence_from_sup["m1_bfaddsentence_None_f1"]))

    # Add the f1 column from the _from_sup dataset to the actual dataset
    # so that we can filter out examples by comparing mask+distractor to supp+distractor
    # instead of just mask+distracto to supp alone
    # ds_got_worse_with_bf_add_sentence = ds_got_worse_with_bf_add_sentence.add_column(
    #     "m1_bfaddsentence_None_f1_from_sup",
    #     ds_got_worse_with_bf_add_sentence_from_sup["m1_bfaddsentence_None_f1"],
    # )
    before = len(ds_got_worse_with_bf_add_sentence)
    ds_got_worse_with_bf_add_sentence = ds_got_worse_with_bf_add_sentence.filter(
        lambda x: x["m1_suppbfaddsentence_None_f1"] - x["m1_bfaddsentence_None_f1"]
        > adversarial_drop_thresh
    )
    print(
        f"filtered out {before - len(ds_got_worse_with_bf_add_sentence)} examples compared to supp+distractor"
    )

    # reduce masked dataset to at most `adversarial_drop_thresh` examples of each `id`
    # also drop examples where the delta between baseline and masked or distracted is less than adversarial_drop_thresh
    # we do nothing with `ds_got_worse_with_bf_add_sentence` because it is already reduced
    # in m1.evaluate
    ds_got_worse_with_bfdelsentence = reduce_to_n(
        ds_got_worse_with_bfdelsentence,
        max_adversarial_examples,
        baseline_f1_col_name="m1_supporting_None_f1",
        exp_f1_col_name="m1_bfdelsentence_None_f1",
        adversarial_drop_thresh=adversarial_drop_thresh,
    )

    ds_got_worse_with_bf_add_sentence = reduce_to_n(
        ds_got_worse_with_bf_add_sentence,
        max_adversarial_examples,
        baseline_f1_col_name="m1_supporting_None_f1",
        exp_f1_col_name="m1_bfaddsentence_None_f1",
        adversarial_drop_thresh=adversarial_drop_thresh,
    )

    output_ds = combine_adversarial_ds(
        ds_got_worse_with_bf_add_sentence, ds_got_worse_with_bfdelsentence
    )

    # df = output_ds.to_pandas()

    # df["id_suffix"] = df["id"].apply(lambda x: x.split("_")[1])
    # df["id"] = df["id"].apply(lambda x: x.split("_")[0])
    return output_ds


def mask_bf_sentence(example, do_single_example=False):
    # new_example = example.copy()
    new_examples = []
    # """Mask random useful sentence in example."""
    n_supporting_facts = len(example["supporting_facts"])
    assert n_supporting_facts > 0, "No supporting facts found"

    # # Locate all the facts
    fact_keys = []
    for i, sentences in enumerate(example["context_supporting"]["sentences"]):
        for j, sentence in enumerate(
            sentences
        ):  # actually len(sentences) = 1 so j is always 0
            fact_keys.append((i, j))
    # shuffle the fact keys
    random.shuffle(fact_keys)

    # create an example for each masked fact
    limit = 1 if do_single_example else len(fact_keys)
    for i in range(limit):
        new_example = deepcopy(example)
        rand_keys = fact_keys[i]
        new_example["masked_sentence"] = new_example["context_supporting"]["sentences"][
            rand_keys[0]
        ][rand_keys[1]]
        new_example["masked_sentence_title"] = new_example["context_None"]["title"][
            rand_keys[0]
        ]
        # Create the context_randomsentence column from everything in the context_None column besides the masked sentence
        new_example["context_bfdelsentence"]["sentences"] = deepcopy(
            new_example["context_supporting"]["sentences"]
        )
        new_example["context_bfdelsentence"]["sentences"][rand_keys[0]].pop(
            rand_keys[1]
        )
        new_examples.append(new_example)

    # create an example for each added distractor
    for i in range(len(example["context_distractor"]["sentences"])):
        new_example = deepcopy(example)
    bf_mini_dataset = Dataset.from_list(new_examples)
    return bf_mini_dataset


def bf_del_sentences(ds, do_single_example=False):
    bf_mini_datasets = [mask_bf_sentence(example, do_single_example) for example in ds]
    new_ds = concatenate_datasets(bf_mini_datasets)
    new_ds = add_flat_contexts(new_ds, ["bfdelsentence"], load_from_cache=False)
    return new_ds


def distract_bf_sentence(example, do_single_example):
    # new_example = example.copy()
    new_examples = []
    n_distractors_sentences = len(example["context_distractor"]["sentences"])
    assert n_distractors_sentences > 0, "No distractor sentences found"

    # # Locate all the distractors
    distractor_keys = []
    for i, sentences in enumerate(example["context_distractor"]["sentences"]):
        for j, sentence in enumerate(
            sentences
        ):  # actually len(sentences) = 1 so j is always 0
            distractor_keys.append((i, j))
    # shuffle the distractor keys
    random.shuffle(distractor_keys)

    # create an example for each distractor
    if do_single_example:
        limit = 1
    else:
        limit = len(distractor_keys)
    for i in range(limit):
        new_example = deepcopy(example)
        # delete the "fc_bfdelsentence" field since it will be out of date.
        # it will be added back byt he add_flat_contexts function
        # del new_example["fc_bfdelsentence"]
        rand_keys = distractor_keys[i]
        new_example["distractor_sentence"] = new_example["context_distractor"][
            "sentences"
        ][rand_keys[0]][rand_keys[1]]

        new_example["context_bfaddsentence"]["sentences"] = deepcopy(
            new_example["context_bfdelsentence"]["sentences"]
            # new_example[src_col]["sentences"]
        )
        new_example["context_bfaddsentence"]["sentences"][rand_keys[0]].insert(
            rand_keys[1], new_example["distractor_sentence"]
        )
        new_example["context_suppbfaddsentence"] = {}
        new_example["context_suppbfaddsentence"]["sentences"] = deepcopy(
            new_example["context_supporting"]["sentences"]
        )
        new_example["context_suppbfaddsentence"]["sentences"][rand_keys[0]].insert(
            rand_keys[1], new_example["distractor_sentence"]
        )
        new_examples.append(new_example)

    # create an example for each added distractor
    for i in range(len(example["context_distractor"]["sentences"])):
        new_example = deepcopy(example)
    bf_mini_dataset = add_flat_contexts(
        Dataset.from_list(new_examples), ["bfaddsentence", "suppbfaddsentence"]
    )

    # drop columns that refer to the bfdelsentence examples from which
    # these examples are derived
    bf_mini_dataset = bf_mini_dataset.remove_columns(
        [
            x
            for x in [
                "m1_bfdelsentence_None_gen",
                "m1_bfdelsentence_None_f1",
                "m1_bfdelsentence_None_em",
                "prepped_bfdelsentence_None",
                "fc_bfdelsentence",
            ]
            if x in bf_mini_dataset.column_names
        ]
    )
    return bf_mini_dataset


def bf_add_sentences(ds, do_single_example):
    """Return a list of datasets where examples in each dataset have a different sentence masked.
    Every example in each dataset will have a different distractor sentence added. If `do_single_example`, run in non-brute force mode and only return a single distracted example
    """
    print("adding distractors...")
    bf_mini_datasets = [
        distract_bf_sentence(example, do_single_example) for example in tqdm(ds)
    ]
    # concatenate and merge dataset who share an id
    tmp_ds = concatenate_datasets(bf_mini_datasets)
    id_dict = dict()

    def add_to_dict(example):
        id = example["id"]
        if id not in id_dict:
            id_dict[id] = []
        id_dict[id].append(example)
        return example

    disable_progress_bar()
    _ = tmp_ds.map(
        add_to_dict,
        batched=False,
    )
    enable_progress_bar()

    output_dss = [Dataset.from_list(id_dict[id]) for id in id_dict]
    return output_dss


def split_distractor(example):
    """Edit the example to remove distractor content. Create a new col containing the distractor content."""

    new_example = example.copy()

    # Sort the titles
    all_titles = example["context_None"]["title"]
    supporting_titles = example["supporting_facts"]["title"]
    # new_example["context_supporting"]["title"] = supporting_titles
    # new_example["context_distractor"]["title"] = [
    # x for x in all_titles if x not in supporting_titles
    # ]

    # Sort the sentences
    new_example["context_supporting"]["sentences"] = [
        [] for _ in range(len(example["context_None"]["sentences"]))
    ]
    new_example["context_distractor"]["sentences"] = [
        [] for _ in range(len(example["context_None"]["sentences"]))
    ]
    # new_example["context_None"]["title"] = example["context_None"]["title"]

    supporting_sentences = defaultdict(set)
    for title, sent_index in zip(
        example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"]
    ):
        supporting_sentences[title].add(sent_index)

    for j, (title, sentences) in enumerate(
        zip(example["context_None"]["title"], example["context_None"]["sentences"])
    ):
        for i, sent in enumerate(sentences):
            if i in supporting_sentences[title]:
                new_example["context_supporting"]["sentences"][j].append(sent)
                # new_example["context_supporting"]["title"].append(title)
            else:
                new_example["context_distractor"]["sentences"][j].append(sent)
                # new_example["context_distractor"]["title"].append(title)
    return new_example


def randsentence_dataset(ds, m1, do_gt):
    # if you use {} instead of {"sentences": []} you encounter a bug in datasets.Dataset.map() version 2.10.1
    ds = ds.add_column(
        "context_bfdelsentence", [{"sentences": []} for _ in range(len(ds))]
    )

    # masking
    # mask one random sentence from each example

    if do_gt:
        bf_mini_datasets = [
            mask_bf_sentence(example).shuffle(seed=0) for example in ds
        ]
    else:
        bf_mini_datasets = [
            mask_bf_sentence(example).shuffle(seed=0).select([0]) for example in ds
        ]
    ds_delsentence = concatenate_datasets(bf_mini_datasets)
    ds_delsentence = add_flat_contexts(
        ds_delsentence, ["bfdelsentence"], load_from_cache=False
    )

    # evaluate the reference col
    output_ds, _metrics = m1.evaluate(
        masking_scheme="bfdelsentence",
        ds=ds_delsentence,
        a2_col=None,
        # max_adversarial_examples=max_adversarial_examples,
        # threshold=-1,
    )

    # rename bfdelsentence -> bf_randsentence

    # df = output_ds.to_pandas()
    # df.rename(columns=col_name_map, inplace=True)
    for k, v in col_name_map.items():
        output_ds = output_ds.rename_column(k, v)
    return output_ds


def randdist_dataset(ds, m1, max_adversarial_examples):
    # if you use {} instead of {"sentences": []} you encounter a bug in datasets.Dataset.map() version 2.10.1
    ds = ds.add_column(
        "context_bfdelsentence", [{"sentences": []} for _ in range(len(ds))]
    )
    ds = ds.add_column(
        "context_bfaddsentence", [{"sentences": []} for _ in range(len(ds))]
    )

    # masking

    ds_bfdelsentence = bf_del_sentences(ds, do_single_example=True)
    # Don't run masking in adversarial mode so that we get a more
    # even distribution once we run it again in distractor mode.
    # print("m1 masking...")
    # ds_bfdelsentence, _metrics = m1.evaluate(
    #     masking_scheme="bfdelsentence", ds=ds_bfdelsentence, a2_col=None
    # )
    # distracting

    mini_dss_bfaddsentence = bf_add_sentences(ds_bfdelsentence, do_single_example=True)

    ds_bfaddsentence = concatenate_datasets(mini_dss_bfaddsentence)

    # reduce masked dataset to at most `adversarial_drop_thresh` examples of each `id`
    # also drop examples where the delta between baseline and masked or distracted is less than adversarial_drop_thresh
    # we do nothing with `ds_got_worse_with_bf_add_sentence` because it is already reduced
    # in m1.evaluate
    df = ds_bfaddsentence.to_pandas()
    # df.rename(columns=col_name_map, inplace=True)
    df.rename(
        columns={
            "fc_bfaddsentence": "fc_masked",
            "context_bfaddsentence": "context_masked",
        },
        inplace=True,
    )
    # delete unused columns
    df = df.drop(
        columns=[
            "fc_suppbfaddsentence",
            "context_suppbfaddsentence",
            "context_bfdelsentence",
        ]
    )
    df_list = list(x[1].head(max_adversarial_examples) for x in df.groupby("id"))
    df = pd.concat(df_list)
    output_ds = Dataset.from_pandas(df)
    output_ds, _metrics = m1.evaluate(
        masking_scheme="masked", ds=output_ds, a2_col=None
    )
    return output_ds


def mask_None(example):
    return example


def reduce_to_n(
    ds: Dataset,
    n: int,
    baseline_f1_col_name: str,
    exp_f1_col_name: str,
    adversarial_drop_thresh: float,
):
    """Reduce the dataset to at most n examples of each `id`, selecting examples with the highest (baseline_f1_col_name - exp_f1_col_name)"""
    df = ds.to_pandas()
    df["delta"] = df[baseline_f1_col_name] - df[exp_f1_col_name]
    df = df[df["delta"] > adversarial_drop_thresh]
    df["rand"] = [random.rand() for _ in range(len(df))]
    # sort by delta, then by rand, then take the top n
    df_list = list(
        x[1].sort_values(by=["delta", "rand"]).head(n) for x in df.groupby("id")
    )
    # df = df.groupby("id").sort_values(by=["delta", "rand"]).head(n)
    df = pd.concat(df_list)
    ds = Dataset.from_pandas(df)
    ds = ds.remove_columns(["__index_level_0__", "delta", "rand"])
    return ds


def retroactively_add_distractors(ds):
    ds = ds.add_column("context_bfdelsentence", ds["context_masked"])
    ds = ds.add_column(
        "context_bfaddsentence", [{"sentences": None} for _ in range(len(ds))]
    )
    ex = distract_bf_sentence(ds[0])
    return ds

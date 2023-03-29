from collections import defaultdict
from numpy import random
from copy import deepcopy
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm


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
    """Convert lists of sentences into single strings under the prefix "fc_. Used both here and in preprocess.py"""
    for masking_scheme in list(masking_schemes):
        masking_str = f"context_{masking_scheme}"
        # print(f"Flattening context {masking_scheme}...")
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
    # print("Normalizing whitespace...")
    # TODO: the list of three should be removed and sent in the call
    for masking_scheme in list(masking_schemes):
        # masking_str = f"context_{masking_scheme}"
        new_ds = new_ds.map(
            lambda x: {
                f"fc_{masking_scheme}": " ".join(x[f"fc_{masking_scheme}"].split())
            },
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache,
        )
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


def mask_random_sentence(example):
    new_example = example.copy()
    """Mask random useful sentence in example."""
    n_supporting_facts = len(new_example["supporting_facts"])
    assert n_supporting_facts > 0, "No supporting facts found"

    # Locate all the facts
    fact_keys = []
    for i, sentences in enumerate(example["context_supporting"]["sentences"]):
        for j, sentence in enumerate(sentences):
            fact_keys.append((i, j))

    # Select one random fact
    rand_index = random.randint(0, len(fact_keys))
    rand_keys = fact_keys[rand_index]
    new_example["masked_sentence"] = new_example["context_supporting"]["sentences"][
        rand_keys[0]
    ][rand_keys[1]]

    # Create the context_randomsentence column from everything in the context_None column besides the masked sentence
    new_example["context_randomsentence"]["sentences"] = deepcopy(
        new_example["context_supporting"]["sentences"]
    )
    new_example["context_randomsentence"]["sentences"][rand_keys[0]].pop(
        rand_keys[1]
    )  # wrong?
    # debug_context = "[SEP]".join(
    #     [" ".join(x) for x in new_example["context_None"]["sentences"]]
    # )
    return new_example
    # return {"masked_col": new_example["context_None"]}


def mask_bf_sentence(example):
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

    # create an example for each masked fact
    for i in range(len(fact_keys)):
        new_example = deepcopy(example)
        rand_keys = fact_keys[i]
        new_example["masked_sentence"] = new_example["context_supporting"]["sentences"][
            rand_keys[0]
        ][rand_keys[1]]
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


def bf_del_sentences(ds):
    bf_mini_datasets = [mask_bf_sentence(example) for example in tqdm(ds)]
    new_ds = concatenate_datasets(bf_mini_datasets)
    new_ds = add_flat_contexts(new_ds, ["bfdelsentence"], load_from_cache=False)
    return new_ds


def distract_bf_sentence(example):
    # new_example = example.copy()
    new_examples = []
    n_distractors_sentences = len(example["context_distractor"]["sentences"])
    assert n_distractors_sentences > 0, "No distractor sentences found"

    # # Locate all the facts
    distractor_keys = []
    for i, sentences in enumerate(example["context_distractor"]["sentences"]):
        for j, sentence in enumerate(
            sentences
        ):  # actually len(sentences) = 1 so j is always 0
            distractor_keys.append((i, j))

    # create an example for each masked fact
    for i in range(len(distractor_keys)):
        new_example = deepcopy(example)
        # delete the "fc_bfdelsentence" field since it will be out of date.
        # it will be added back byt he add_flat_contexts function
        # del new_example["fc_bfdelsentence"]
        rand_keys = distractor_keys[i]
        new_example["distractor_sentence"] = new_example["context_distractor"][
            "sentences"
        ][rand_keys[0]][rand_keys[1]]
        # Create the context_randomsentence column from everything in the context_None column besides the masked sentence
        new_example["context_bfaddsentence"]["sentences"] = deepcopy(
            new_example["context_supporting"]["sentences"]
        )
        new_example["context_bfaddsentence"]["sentences"][rand_keys[0]].insert(
            rand_keys[1], new_example["distractor_sentence"]
        )
        new_example["is_distracted"] = True
        new_examples.append(new_example)

    # create an example for each added distractor
    for i in range(len(example["context_distractor"]["sentences"])):
        new_example = deepcopy(example)
    bf_mini_dataset = add_flat_contexts(
        Dataset.from_list(new_examples), ["bfaddsentence"]
    )
    return bf_mini_dataset


def bf_add_sentences(ds):
    bf_mini_datasets = [distract_bf_sentence(example) for example in tqdm(ds)]
    new_ds = concatenate_datasets(bf_mini_datasets)
    return new_ds


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


def mask_None(example):
    return example

def reduce_to_n(ds, n):
    """Reduce the dataset to at most n examples of each `id`"""
    df = ds.to_pandas()
    df = df.groupby("id").head(n)
    return Dataset.from_pandas(df)
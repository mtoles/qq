from collections import defaultdict
from numpy import random
from copy import deepcopy
from datasets import Dataset


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

    # # Select one random fact
    # rand_index = random.randint(0, len(fact_keys))
    # rand_keys = fact_keys[rand_index]
    # new_example["masked_sentence"] = new_example["context_supporting"]["sentences"][
    #     rand_keys[0]
    # ][rand_keys[1]]

    # create an example for each masked fact
    for i in range(len(fact_keys)):
        new_example = deepcopy(example)
        rand_keys = fact_keys[i]
        new_example["masked_sentence"] = new_example["context_supporting"]["sentences"][
            rand_keys[0]
        ][rand_keys[1]]
        # Create the context_randomsentence column from everything in the context_None column besides the masked sentence
        new_example["context_bfsentence"]["sentences"] = deepcopy(
            new_example["context_supporting"]["sentences"]
        )
        new_example["context_bfsentence"]["sentences"][rand_keys[0]].pop(rand_keys[1])
        new_examples.append(new_example)

    # create an example for each added distractor
    for i in range(len(example["context_distractor"]["sentences"])):
        new_example = deepcopy(example)
        

    bf_mini_dataset = Dataset.from_list(new_examples)

    # Run M1 on the dataset
    # m1.evaluate("bfsentence", bf_mini_dataset, None)

    # # Create the context_randomsentence column from everything in the context_None column besides the masked sentence
    # new_example["context_randomsentence"]["sentences"] = deepcopy(
    #     new_example["context_supporting"]["sentences"]
    # )
    # new_example["context_randomsentence"]["sentences"][rand_keys[0]].pop(rand_keys[1])
    return bf_mini_dataset

def mask_bf_sentences(ds):
    

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

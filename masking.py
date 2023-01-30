from collections import defaultdict
from numpy import random


def mask_random_sentence(example):
    # TODO: mask the input ids instead of the context
    """Mask random useful sentence in example. Return the column entry, not the whole example."""
    titles = example["context_None"]["title"]
    # create a dictionary mapping each title to its index
    title_to_index = {title: i for i, title in enumerate(titles)}

    n_supporting_facts = len(example["supporting_facts"])
    assert n_supporting_facts > 0, "No supporting facts found"

    # randomly select a supporting fact
    i = random.randint(0, n_supporting_facts - 1)
    fact_title = example["supporting_facts"]["title"][i]
    fact_title_index = title_to_index[fact_title]
    fact_sent_index = example["supporting_facts"]["sent_id"][i]
    fact_sent = example["context_None"]["sentences"][fact_title_index][fact_sent_index]
    len_fact = len(fact_sent.split())
    # replacement = " ".join(["[MASK]"] * len_fact)
    replacement = ""

    example["context_None"]["sentences"][fact_title_index][
        fact_sent_index
    ] = replacement

    debug_context = "[SEP]".join(
        [" ".join(x) for x in example["context_None"]["sentences"]]
    )
    return {"masked_col": example["context_None"]}
    # return example


def split_distractor(example):
    """Edit the example to remove distractor content. Create a new col containing the distractor content."""

    new_example = example.copy()

    # Sort the titles
    all_titles = example["context_None"]["title"]
    supporting_titles = example["supporting_facts"]["title"]
    new_example["context_None"]["title"] = supporting_titles
    new_example["context_distractor"]["title"] = [
        x for x in all_titles if x not in supporting_titles
    ]

    # Sort the sentences
    new_example["context_None"]["sentences"] = [
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
                new_example["context_None"]["sentences"][j].append(sent)
                new_example["context_None"]["title"].append(title)
            else:
                new_example["context_distractor"]["sentences"][j].append(sent)
                new_example["context_distractor"]["title"].append(title)
    return new_example

    # num_titles = len(example["context_None"]["title"])
    # for i in reversed(range(num_titles)):
    #     if all_titles[i] not in supporting_titles:
    #         example["context_None"]["sentences"].pop(i)
    #         example["context_None"]["title"].pop(i)

    return example


def mask_None(example):
    return example

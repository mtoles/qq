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
    replacement = " ".join(["[MASK]"] * len_fact)

    example["context_None"]["sentences"][fact_title_index][
        fact_sent_index
    ] = replacement

    debug_context = "[SEP]".join(
        [" ".join(x) for x in example["context_None"]["sentences"]]
    )
    return {"masked_col": example["context_None"]}
    # return example


def drop_distractor(example):
    """Edit the example IN PLACE to remove distractor content."""
    supporting_titles = example["supporting_facts"]["title"]
    all_titles = example["context_None"]["title"]
    for i in reversed(
        range(len(example["context_None"]["title"]))
    ):  # fix edit while iterating issue
        if all_titles[i] not in supporting_titles:
            example["context_None"]["sentences"].pop(i)  # TODO: iterate backwards
            example["context_None"]["title"].pop(i)
    return example


def mask_None(example):
    return example

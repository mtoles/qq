"""
Utilities for processing datasets
"""


PUNCTUATION_SET_TO_EXCLUDE = set("".join(["‘", "’", "´", "`", ".", ",", "-", '"']))


def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]


def expand_to_aliases(given_answers, make_sub_answers=False):
    if make_sub_answers:
        # if answers are longer than one word, make sure a predictions is correct if it coresponds to the complete 1: or :-1 sub word
        # *e.g.* if the correct answer contains a prefix such as "the", or "a"
        given_answers = (
            given_answers
            + get_sub_answers(given_answers, begin=1)
            + get_sub_answers(given_answers, end=-1)
        )
    answers = []
    for answer in given_answers:
        alias = answer.replace("_", " ").lower()
        alias = "".join(
            c if c not in PUNCTUATION_SET_TO_EXCLUDE else " " for c in alias
        )
        answers.append(" ".join(alias.split()).strip())
    return set(answers)


# dataset formatting
def format_dataset_hotpot(example):
    # the context might be comprised of multiple contexts => me merge them here
    example["context"] = "\n".join(
        [x for y in example["context"]["sentences"] for x in y]
    )
    # example["targets"] = example["answer"]["aliases"]
    # example["norm_target"] = example["answer"]["normalized_value"]
    example["targets"] = [example["answer"]]
    return example


def format_dataset_trivia(example):
    # the context might be comprised of multiple contexts => me merge them here
    example["context"] = " ".join(
        ("\n".join(example["entity_pages"]["wiki_context"])).split("\n")
    )
    example["targets"] = example["answer"]["aliases"]
    example["norm_target"] = example["answer"]["normalized_value"]
    return example


def has_answer(example, masking_str):
    is_answerable = (example["answer"] in ["yes", "no"]) or (
        example["answer"].lower() in example[masking_str].lower()
    )
    return is_answerable


def drop_unanswerable(dataset, masking_scheme, load_from_cache_file):
    masking_str = f"fc_{masking_scheme}"
    clean_ds = dataset.filter(lambda x: has_answer(x, masking_str), load_from_cache_file=load_from_cache_file)
    print(
        f"dropped {len(dataset) - len(clean_ds)}/{len(dataset)} unanswerable examples"
    )
    return clean_ds

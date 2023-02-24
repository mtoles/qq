"""
Utilities for processing datasets
"""
from hotpot_evaluate_v1 import f1_score, normalize_answer
from utils import sublist_is_in_list

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
    answer = [normalize_answer(x) for x in example["answer"].split()]
    context = [normalize_answer(x) for x in example[masking_str].split()]
    if answer[0] in ["yes", "no"]:
        return True
    if sublist_is_in_list(answer, context):
        return True

    return False


def drop_unanswerable(dataset, masking_scheme, load_from_cache_file):
    print("dropping unanswerable examples...")
    masking_str = f"fc_{masking_scheme}"
    clean_ds = dataset.filter(
        lambda x: has_answer(x, masking_str), load_from_cache_file=load_from_cache_file
    )
    print(
        f"dropped {len(dataset) - len(clean_ds)}/{len(dataset)} unanswerable examples"
    )
    return clean_ds


def clean_answer(ex, tk):
    ex["answer"] = tk.decode(tk.encode(ex["answer"]))
    return ex


def check_example(ex, tk):
    st = ex["labels"]["start_token"][0]
    et = ex["labels"]["end_token"][0]
    input_ids = ex["input_ids"]
    answer = ex["answer"]
    answer_tokens = input_ids[st : et + 1]
    answer_indexed = tk.decode(answer_tokens)
    # assert (
    #     answer == answer_indexed
    # ), f"answer {answer} != {answer_indexed} at {st}:{et}"
    if st == -100 and et == -100:
        if answer not in ["yes", "no"]:
            print(f"answer {answer} should be 'yes' or 'no' if st and et are -100")
    else:
        # run the answer through the tokenizer so it doesn't trigger on tokenizer failures
        # since the input_ids are tokenized elsewhere
        tk_answer = tk.decode(tk.encode(answer)[1:-1])
        f1, precision, recall = f1_score(tk_answer, answer_indexed)
        if not (f1 == 1 and precision == 1 and recall == 1):
            print(f"answer {tk_answer} != {answer_indexed} at {st}:{et}")


def check_dataset(dataset, tk):
    """Check that answers are actually at their identified position in the context and other sanity checks"""
    print("Checking dataset...")
    dataset.map(
        lambda x: check_example(x, tk),
        batched=False,
        load_from_cache_file=False,
    )

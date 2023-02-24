"""Process and tokenize the data from preprocess.py at training and validation time"""

import os

from typing import List, Optional, Tuple
from collections import defaultdict
from hotpot_evaluate_v1 import normalize_answer
from utils import (
    find_sublist_in_list,
    CATEGORY_MAPPING,
)

DOC_STRIDE = 2048
MAX_LENGTH = 4096
SEED = 42
PROCESS_TRAIN = os.environ.pop("PROCESS_TRAIN", "false")


def _get_single_answer_data(example):
    def choose_first(answer, is_long_answer=False):
        assert isinstance(answer, list)
        assert len(answer) == 1  # matt
        if len(answer) == 1:
            answer = answer[0]
            return {k: [answer[k]] for k in answer} if is_long_answer else answer
        for a in answer:
            if is_long_answer:
                a = {k: [a[k]] for k in a}
            if len(a["start_token"]) > 0:
                break
        return a

    answer = {"id": example["id"]}
    annotation = example["annotations"]
    yes_no_answer = annotation["yes_no_answer"]
    if 0 in yes_no_answer or 1 in yes_no_answer:
        answer["category"] = ["yes"] if 1 in yes_no_answer else ["no"]
        answer["start_token"] = answer["end_token"] = []
        answer["start_byte"] = answer["end_byte"] = []
        answer["text"] = ["<cls>"]
    else:
        answer["category"] = ["short"]
        out = choose_first(annotation["short_answers"])
        if len(out["start_token"]) == 0:
            # answer will be long if short is not available
            answer["category"] = ["long"]
            out = choose_first(annotation["long_answer"], is_long_answer=True)
            out["text"] = []
        answer.update(out)

    cols = ["start_token", "end_token", "start_byte", "end_byte", "text"]
    if not all([isinstance(answer[k], list) for k in cols]):
        raise ValueError("Issue in ID", example["id"])

    return answer


def get_context_and_ans(example, masking_scheme):
    """Gives new context after removing <html> & new answer tokens as per new context"""
    answer = _get_single_answer_data(example)
    # bytes are of no use
    del answer["start_byte"]
    del answer["end_byte"]

    # handle yes_no answers explicitly
    if answer["category"][0] in ["yes", "no"]:  # category is list with one element
        doc = example["document"]["tokens"]
        context = []
        for i in range(len(doc["token"])):
            if not doc["is_html"][i]:
                context.append(doc["token"][i])
        return {
            "context": " ".join(context),
            "answer": {
                "start_token": -100,  # ignore index in cross-entropy # TODO: Why not -1 like elsewhere?
                "end_token": -100,  # ignore index in cross-entropy
                "category": answer["category"],
                "span": answer["category"],  # extra
            },
        }

    # handling normal samples

    cols = ["start_token", "end_token"]
    answer.update(
        {k: answer[k][0] if len(answer[k]) > 0 else answer[k] for k in cols}
    )  # e.g. [10] == 10

    doc = example["document"]["tokens"]
    start_token = answer["start_token"]
    end_token = answer["end_token"]

    context = []
    for i in range(len(doc["token"])):
        if not doc["is_html"][i]:
            context.append(doc["token"][i])
        else:
            if answer["start_token"] > i:
                start_token -= 1
            if answer["end_token"] > i:
                end_token -= 1
    new = " ".join(context[start_token:end_token])

    output = {
        "context": " ".join(context),
        "answer": {
            "start_token": start_token,
            "end_token": end_token - 1,  # this makes it inclusive
            "category": answer["category"],  # either long or short
            "span": new,  # extra
        },
    }
    return output


def get_strided_contexts_and_ans(
    example,
    tk,
    max_length,
    masking_scheme,
):
    # overlap will be of doc_stride - q_len

    out = get_context_and_ans(example, masking_scheme)
    answer = out["answer"]
    input_ids = tk(out["context"]).input_ids

    # later, removing these samples
    if answer["start_token"] == -1:
        return {
            "example_id": example["id"],
            "input_ids": input_ids,
            "labels": {
                "start_token": [-1],
                "end_token": [-1],
                "category": answer["category"],
            },
        }

    # q_len = input_ids.index(tokenizer.sep_token_id) + 1

    # return yes/no
    if answer["category"][0] in ["yes", "no"]:  # category is list with one element
        assert (
            len(input_ids) < max_length
        ), "input_ids should be greater than max_length"

        output = {
            "example_id": example["id"],
            "input_ids": input_ids,
            "labels": {
                "start_token": [-100],
                "end_token": [-100],
                "category": answer["category"],
            },
        }
    # question is not a yes/no question
    else:
        splitted_context = out["context"].split()
        complete_end_token = splitted_context[answer["end_token"]]
        answer["start_token"] = (
            len(
                tk(
                    " ".join(splitted_context[: answer["start_token"]]),
                    add_special_tokens=False,
                ).input_ids
            )
            + 1
        )
        answer["end_token"] = (
            len(
                tk(
                    " ".join(splitted_context[: answer["end_token"]]),
                    add_special_tokens=False,
                ).input_ids
            )
            + 1
        )

        answer["start_token"]  # += q_len
        answer["end_token"]  # += q_len

        # fixing end token
        num_sub_tokens = len(tk(complete_end_token, add_special_tokens=False).input_ids)
        if num_sub_tokens > 1:
            answer["end_token"] += num_sub_tokens - 1

        old = input_ids[
            answer["start_token"] : answer["end_token"] + 1
        ]  # right & left are inclusive
        start_token = answer["start_token"]
        end_token = answer["end_token"]

        # input is short enough to fit inside max_length
        if len(input_ids) <= max_length:
            output = {
                "example_id": example["id"],
                "input_ids": input_ids,
                "labels": {
                    "start_token": [answer["start_token"]],
                    "end_token": [answer["end_token"]],
                    "category": answer["category"],
                },
            }
        # input is longer than max_length
        else:
            print("LONGER THAN MAX LENGTH")
            raise NotImplementedError
    return output


# def prepare_inputs_nq(example, tokenizer, doc_stride=2048, max_length=4096):
#     example = get_strided_contexts_and_ans(
#         example,
#         tokenizer,
#         doc_stride=doc_stride,
#         max_length=max_length,
#     )

#     return example


def prepare_inputs_hp(
    example,
    tk,
    max_length,
    masking_scheme,
):
    adapted_example = adapt_example(example, masking_scheme=masking_scheme)
    tokenized_example = get_strided_contexts_and_ans(
        adapted_example,
        tk,
        max_length=max_length,
        masking_scheme=masking_scheme,
    )

    return tokenized_example


def get_answer_token_indices(context: str, answer: str) -> Tuple[int, int]:
    if answer in ["yes", "no"]:
        return -1, -1
    context = " ".join(context.split())  # normalize whitespaces
    context_li = [normalize_answer(x) for x in context.split()]
    answer_li = [normalize_answer(x) for x in answer.split()]
    start_token_index = find_sublist_in_list(answer_li, context_li)
    end_token_index = start_token_index + len(answer_li)
    return start_token_index, end_token_index


def adapt_example(example, masking_scheme=None):
    masking_scheme = str(masking_scheme)
    masking_str = (
        f"prepped_{masking_scheme}"  # operate on the prepped text, not the fc text,
    )
    """Convert the HP example to look like an NQ example"""
    new_example = {}
    new_example["question"] = {"text": example["question"]}
    answer = example["answer"]
    new_example["answer"] = {"text": answer}
    # Add the question to the context
    context = example[masking_str]
    # Call join/split an extra time to normalize whitespaces and unicode nonsense
    context = " ".join(context.split())
    answer = " ".join(answer.split())
    tokens = context.split()
    new_example["document"] = {
        "html": context,
        "tokens": {"token": tokens, "is_html": [False for _ in tokens]},
    }
    new_example["id"] = example["id"]
    start_token_index, end_token_index = get_answer_token_indices(context, answer)
    yn_dict = defaultdict(lambda: [-1])
    yn_dict["yes"] = [1]
    yn_dict["no"] = [0]
    new_example["annotations"] = {
        "short_answers": [
            {
                "start_token": [start_token_index],
                "end_token": [end_token_index],
                "start_byte": [-1],  # should be unused
                "end_byte": [-1],  # should be unused
                "text": [answer],
            }
        ],
        "yes_no_answer": yn_dict[answer],
    }
    # assert answer in ["yes", "no"] or normalize_answer(
    #     example["answer"]
    # ) == normalize_answer(" ".join(tokens[start_token_index:end_token_index]))
    return new_example


def prepend_question(example, masking_scheme, sep_token):
    # masking_str = f"fc_{masking_scheme}"
    context = example[f"prepped_{masking_scheme}"]
    question = example["question"]
    example[f"prepped_{masking_scheme}"] = " ".join(
        " ".join([question, sep_token, context]).split()
    )
    return example

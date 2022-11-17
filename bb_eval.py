# Code borrowed from
# https://colab.research.google.com/drive/1DVOm1VHjW0eKCayFq1N2GpY6GR9M4tJP?usp=sharing#scrollTo=sHgcdBt41gby
# https://github.com/thevasudevgupta/bigbird/blob/main/LICENSE
# Used under MIT License

# %%
from transformers import (
    BigBirdConfig,
    BigBirdForQuestionAnswering,
    BigBirdTokenizer,
    AutoTokenizer,
)
import datasets
from datasets import load_dataset
import torch

# import pickle as pkl


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
    example["targets"] = example["answer"]
    return example


def format_dataset_trivia(example):
    # the context might be comprised of multiple contexts => me merge them here
    # example["context"] = "\n".join([x for y in example['context']['sentences'] for x in y])
    example["targets"] = example["answer"]["aliases"]
    example["norm_target"] = example["answer"]["normalized_value"]
    example["targets"] = example["answer"]
    return example


def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]


def evaluate(example):
    # encode question and context so that they are seperated by a tokenizer.sep_token and cut at max_length
    encoding = tk(
        example["question"],
        example["context"],
        return_tensors="pt",
        max_length=4096,
        padding="max_length",
        truncation=True,
    )
    input_ids = encoding.input_ids.to("cuda")

    with torch.no_grad():
        start_scores, end_scores = model(input_ids=input_ids).to_tuple()

    start_score, end_score = get_best_valid_start_end_idx(
        start_scores[0], end_scores[0], top_k=8, max_size=16
    )

    n = len(input_ids)
    example["output"] = []
    example["match"] = []
    for i in range(n):
        # Let's convert the input ids back to actual tokens
        all_tokens = tk.convert_ids_to_tokens(encoding["input_ids"][i].tolist())
        answer_tokens = all_tokens[start_score : end_score + 1]

        example["output"].append(tk.decode(tk.convert_tokens_to_ids(answer_tokens)))
        # .replace('"', '')  # remove space prepending space token and remove unnecessary '"'

        answers = expand_to_aliases([example["targets"][i]], make_sub_answers=True)
        predictions = expand_to_aliases([example["output"][i]])

        # if there is a common element, it's a match
        example["match"].append(len(list(answers & predictions)) > 0)

    return example


hotpot_dataset = load_dataset("hotpot_qa", "fullwiki", split="validation[:100]").map(
    format_dataset_hotpot
)
model_id = "google/bigbird-base-trivia-itc"
model = BigBirdForQuestionAnswering.from_pretrained(
    model_id, attention_type="original_full"
).cuda()
tk = BigBirdTokenizer.from_pretrained(model_id)

PUNCTUATION_SET_TO_EXCLUDE = set("".join(["‘", "’", "´", "`", ".", ",", "-", '"']))

# print(evaluate(hotpot_dataset[0]))
results_ds = hotpot_dataset.map(evaluate, batched=True, batch_size=4)  # todo: enable batching
print(results_ds["match"].count(True) / len(results_ds["match"]))
pass

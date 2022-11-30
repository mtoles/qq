# Code borrowed from
# https://colab.research.google.com/drive/1DVOm1VHjW0eKCayFq1N2GpY6GR9M4tJP?usp=sharing#scrollTo=sHgcdBt41gby
# https://github.com/thevasudevgupta/bigbird/blob/main/LICENSE
# Used under MIT License

# %%
from dataset_utils import *

from transformers import (
    BigBirdForQuestionAnswering,
    BigBirdTokenizer,
)
from datasets import load_dataset
import datasets
import torch
import click


def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]


def evaluate(example, model, tk):
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


    n = len(input_ids)
    example["output"] = []
    example["match"] = []
    for i in range(n):
        start_score, end_score = get_best_valid_start_end_idx(
            start_scores[i], end_scores[i], top_k=8, max_size=16
        )
        # Let's convert the input ids back to actual tokens
        all_tokens = tk.convert_ids_to_tokens(encoding["input_ids"][i].tolist())
        answer_tokens = all_tokens[start_score : end_score + 1]

        example["output"].append(tk.decode(tk.convert_tokens_to_ids(answer_tokens)))
        # .replace('"', '')  # remove space prepending space token and remove unnecessary '"'

        answers = expand_to_aliases(example["targets"][i], make_sub_answers=True)
        predictions = expand_to_aliases([example["output"][i]])

        # if there is a common element, it's a match
        example["match"].append(len(list(answers & predictions)) > 0)

    return example


@click.command()
@click.option("--model", help="Model name. Implemented models: {bigbird}")
@click.option(
    "--attention_type",
    default=None,
    help="attention type used for Big Bird: {block_sparse | original_full}",
)
@click.option("--dataset_name", help="Dataset name. Implemented datasets: {hotpot}")
@click.option("--batch_size", default=4, help="Batch size")
@click.option(
    "--downsample_data_size", default=None, help="use at most this many examples"
)
@click.option(
    "--load_from_cache",
    default=True,
    help="Disable to prevent loading from any cache (i.e. hugging face datasets, .map",
)
def main(
    model,
    attention_type,
    dataset_name,
    batch_size,
    downsample_data_size,
    load_from_cache,
):
    if downsample_data_size is not None:
        downsample_str = f"[:{downsample_data_size}]"
    else:
        downsample_str = ""
    if dataset_name == "hotpot":
        dataset = load_dataset(
            "hotpot_qa", "fullwiki", split=f"validation{downsample_str}"
        ).map(format_dataset_hotpot, load_from_cache_file=load_from_cache)
    elif dataset_name == "trivia":
        dataset = datasets.load_dataset(
            "trivia_qa", "rc", split=f"validation{downsample_str}"
        ).map(
            format_dataset_trivia,
            load_from_cache_file=load_from_cache,
            remove_columns=[
                "search_results",
                "question_source",
                "entity_pages",
                "question_id",
            ],
        )
        # filter examples with no context
        dataset = dataset.filter(lambda x: len(x["context"]) > 0)
        # filter examples that are too long
        dataset = dataset.filter(
            lambda x: (len(x["question"]) + len(x["context"])) < 4 * 4096
        )
        # filter examples where the answer is not contained in the context
        # dataset = dataset.filter(lambda x: x["answer"]["value"] in x["context"])

    else:
        raise ValueError("Dataset not implemented")
    if model == "bigbird":
        assert attention_type in [
            "block_sparse",
            "original_full",
        ], f"invalid attention_type: {attention_type}"
        model_id = "google/bigbird-base-trivia-itc"
        model = BigBirdForQuestionAnswering.from_pretrained(
            model_id, attention_type=attention_type
        ).cuda()
        tk = BigBirdTokenizer.from_pretrained(model_id)
    else:
        raise ValueError("Model not implemented")

    # print(evaluate(hotpot_dataset[0]))
    results_ds = dataset.map(
        lambda x: evaluate(x, model, tk), batched=True, batch_size=batch_size
    )  # todo: enable batching
    print("accuracy: ", results_ds["match"].count(True) / len(results_ds["match"]))
    pass


if __name__ == "__main__":
    main()

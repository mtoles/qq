# Code borrowed from
# https://colab.research.google.com/drive/1DVOm1VHjW0eKCayFq1N2GpY6GR9M4tJP?usp=sharing#scrollTo=sHgcdBt41gby
# https://github.com/thevasudevgupta/bigbird/blob/main/LICENSE
# Used under MIT License

# %%
from dataset_utils import *

from transformers import BigBirdTokenizer

from datasets import load_dataset
import datasets
import torch
import click
import pandas as pd

from prepare_data import CATEGORY_MAPPING
from bb_model import BigBirdForNaturalQuestions

INVERSE_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}


def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]


def evaluate(example, model, tk):
    input_ids = torch.Tensor(example["input_ids"]).int().unsqueeze(dim=0).to("cuda")

    with torch.no_grad():
        # start_scores, end_scores = model(input_ids=input_ids).to_tuple()
        model_output = model(input_ids=input_ids)

    loss = model_output["loss"]
    start_scores = model_output["start_logits"]
    end_scores = model_output["end_logits"]
    cls_out = model_output["cls_out"].argmax()

    cat_pred = INVERSE_CATEGORY_MAPPING[cls_out.item()]

    n = len(input_ids)
    example["output"] = []
    example["match"] = []

    start_idx_pred, end_idx_pred = get_best_valid_start_end_idx(
        start_scores[0], end_scores[0], top_k=8, max_size=16
    )
    # Let's convert the input ids back to actual tokens
    # all_tokens = tk.convert_ids_to_tokens(encoding["input_ids"][i].tolist())
    # answer_tokens = all_tokens[start_score : end_score + 1]
    answer_tokens_pred = example["input_ids"][start_idx_pred : end_idx_pred + 1]

    # get ground truth answers
    category = INVERSE_CATEGORY_MAPPING[example["category"]]
    if category in ["yes", "no"]:
        answer_gt = set([category])
    else:
        # .replace('"', '')  # remove space prepending space token and remove unnecessary '"'
        start_idx_gt = example["start_token"]
        end_idx_gt = example["end_token"]
        # answer = example["input_ids"][st:et]

        answer_gt = expand_to_aliases(
            [tk.decode(input_ids[0][start_idx_gt : end_idx_gt + 1])],
            make_sub_answers=True,
        )

    # get predicted answers
    if cat_pred in ["yes", "no"]:
        example["output"] = cat_pred
    else:
        example["output"] = tk.decode(answer_tokens_pred)

    predictions = expand_to_aliases([example["output"]])

    # if there is a common element, it's a match
    example["match"].append(len(list(answer_gt & predictions)) > 0)

    # print("start, end: ", start_idx_pred, end_idx_pred)
    # print("predictions : ", example["output"])
    # print("ground truth: ", answer_gt)
    # print("match: ", example["match"])
    # o = example["output"][i]
    # a = example["answer"][i]
    # q = example["question"][i]
    # c = example["context"][i]

    return example


@click.command()
@click.option("--model_path", default=None, help="Path to model")
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
@click.option("--cache_dir", default=None, help="Preprocessed data directory")
def main(
    model_path,
    model,
    attention_type,
    dataset_name,
    batch_size,
    downsample_data_size,
    load_from_cache,
    cache_dir,
):
    if downsample_data_size is not None:
        downsample_str = f"[:{downsample_data_size}]"
    else:
        downsample_str = ""
    if dataset_name == "hotpot":
        # dataset = load_dataset(
        #     "hotpot_qa", "fullwiki", split=f"validation{downsample_str}"
        # ).map(format_dataset_hotpot, load_from_cache_file=load_from_cache)
        dataset = load_dataset("json", data_files="data/hotpot-validation.jsonl")[
            "train"
        ]
    # elif dataset_name == "trivia":
    #     dataset = datasets.load_dataset(
    #         "trivia_qa", "rc", split=f"validation{downsample_str}"
    #     ).map(
    #         format_dataset_trivia,
    #         load_from_cache_file=load_from_cache,
    #         remove_columns=[
    #             "search_results",
    #             "question_source",
    #             "entity_pages",
    #             "question_id",
    #         ],
    #     )
    #     # filter examples with no context
    #     dataset = dataset.filter(lambda x: len(x["context"]) > 0)
    #     # filter examples that are too long
    #     dataset = dataset.filter(
    #         lambda x: (len(x["question"]) + len(x["context"])) < 4 * 4096
    #     )
    #     # filter examples where the answer is not contained in the context
    #     # dataset = dataset.filter(lambda x: x["answer"]["value"] in x["context"])

    else:
        raise ValueError("Dataset not implemented")
    if model == "bigbird":
        assert attention_type in [
            "block_sparse",
            "original_full",
        ], f"invalid attention_type: {attention_type}"
        model_id = "google/bigbird-base-trivia-itc"
        if model_path is None:
            model = BigBirdForNaturalQuestions.from_pretrained(
                model_id, attention_type=attention_type
            ).cuda()
        else:
            model = BigBirdForNaturalQuestions.from_pretrained(
                model_path, attention_type=attention_type
            ).cuda()
        tk = BigBirdTokenizer.from_pretrained(model_id)
    else:
        raise ValueError("Model not implemented")

    # TESTING
    # dataset = dataset.train_test_split(test_size=100)["test"]

    results_df = pd.DataFrame(dataset)
    matches = [
        x[0]
        for x in dataset.map(
            lambda x: evaluate(x, model, tk), batched=False, load_from_cache_file=False
        )["match"]
    ]
    results_df["match"] = matches
    print("accuracy: ", len(results_df[results_df["match"] == True]) / len(results_df))
    pass


if __name__ == "__main__":
    main()

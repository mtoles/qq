# Code borrowed from
# https://colab.research.google.com/drive/1DVOm1VHjW0eKCayFq1N2GpY6GR9M4tJP?usp=sharing#scrollTo=sHgcdBt41gby
# https://github.com/thevasudevgupta/bigbird/blob/main/LICENSE
# Used under MIT License

# %%

from transformers import BigBirdTokenizer

from datasets import load_dataset
import datasets
import torch
import click
import pandas as pd

from prepare_data import CATEGORY_MAPPING
from bb_model import BigBirdForNaturalQuestions, _get_metrics_single


def evaluate(example, model, tk):
    input_ids = torch.Tensor(example["input_ids"]).int().unsqueeze(dim=0).to("cuda")

    with torch.no_grad():
        # start_scores, end_scores = model(input_ids=input_ids).to_tuple()
        model_output = model(input_ids=input_ids)

    start_logits = model_output["start_logits"]
    end_logits = model_output["end_logits"]
    cls_logits = model_output["cls_out"]

    start_idx_gt = example["start_token"]
    end_idx_gt = example["end_token"]
    cls_gt = example["category"]

    accuracy = _get_metrics_single(
        start_logits,
        end_logits,
        cls_logits,
        input_ids,
        start_idx_gt,
        end_idx_gt,
        cls_gt,
        tk,
    )

    return accuracy


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

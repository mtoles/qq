from transformers import BigBirdTokenizer
from sklearn.model_selection import train_test_split
from pathlib import PurePath

MODEL_ID = "google/bigbird-base-trivia-itc"
tokenizer = BigBirdTokenizer.from_pretrained(MODEL_ID)


def get_downsample_dataset_size_str(downsample_data_size):
    if downsample_data_size is not None:
        downsample_str = f"[:{downsample_data_size}]"
    else:
        downsample_str = ""
    return downsample_str


def make_cache_file_name(split, dataset, downsample_data_size, masking_scheme):
    masking_scheme = str(masking_scheme).replace("_", "")
    cache_file_name = (
        PurePath("data") / f"{dataset}-{split}-{downsample_data_size}-{masking_scheme}"
    )
    return str(cache_file_name)

def dc(input_ids):
    return tokenizer.decode(input_ids)
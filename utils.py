import torch
import torch.nn.functional as F

from transformers import BigBirdTokenizer
from sklearn.model_selection import train_test_split
from pathlib import PurePath

BB_MODEL_ID = "google/bigbird-base-trivia-itc"
GPT_NEO_X_MODEL_ID = "EleutherAI/gpt-neo-20B"
CATEGORY_MAPPING = {"null": 0, "short": 1, "long": 2, "yes": 3, "no": 4}
INVERSE_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}
PAD_ID = -100


def get_downsample_dataset_size_str(downsample_data_size):
    if downsample_data_size is not None:
        downsample_str = f"[:{downsample_data_size}]"
    else:
        downsample_str = ""
    return downsample_str


def make_cache_file_name(split, dataset, downsample_data_size, masking_schemes):
    masking_scheme = "".join(masking_schemes)
    cache_file_name = (
        PurePath("data") / f"{dataset}-{split}-{downsample_data_size}-{masking_scheme}"
    )
    return str(cache_file_name)


def stack_with_padding(tensor_list):
    max_len = max([t.shape[0] for t in tensor_list])
    padded_tensor_list = []
    for i in range(len(tensor_list)):
        container = torch.zeros(max_len, dtype=tensor_list[i].dtype) + PAD_ID
        container[: tensor_list[i].shape[0]] = tensor_list[i]
        padded_tensor_list.append(container)
    output = torch.stack(padded_tensor_list, dim=0)
    return output



def unstack_with_padding(tensor, lengths):
    return [t[:l] for t, l in zip(tensor, lengths)]

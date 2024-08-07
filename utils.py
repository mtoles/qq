import torch
import os
from sklearn.model_selection import train_test_split
import random
import numpy as np

BB_MODEL_ID = "google/bigbird-base-trivia-itc"
GPT_NEO_MODEL_ID = "EleutherAI/gpt-neo-2.7B"
CATEGORY_MAPPING = {"null": 0, "short": 1, "long": 2, "yes": 3, "no": 4}
INVERSE_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def get_downsample_dataset_size_str(downsample_data_size):
    if str(downsample_data_size) != "None":
        downsample_str = f"[:{downsample_data_size}]"
    else:
        downsample_str = ""
    return downsample_str


def make_cache_file_name(
    dataset, split, downsample_data_size, masking_schemes, distract_or_focus
):
    cache_file_name = f"{dataset}-{split}-{downsample_data_size}-{''.join(list(masking_schemes)+['None'])}-{distract_or_focus}"
    return cache_file_name


def stack_with_padding(tensor_list, pad_id):
    max_len = max([t.shape[0] for t in tensor_list])
    padded_tensor_list = []
    for i in range(len(tensor_list)):
        container = torch.zeros(max_len, dtype=tensor_list[i].dtype) + pad_id
        container[: tensor_list[i].shape[0]] = tensor_list[i]
        padded_tensor_list.append(container)
    output = torch.stack(padded_tensor_list, dim=0)
    return output


def unpad_and_tokenize(tokenized_answer, tk):
    output = []
    tokenized_answer = standardize_padding(tokenized_answer)
    for ta in tokenized_answer:
        output.append(tk.decode(ta))
    return output


def unstack_with_padding(tensor, lengths):
    return [t[:l] for t, l in zip(tensor, lengths)]


def collate_fn_bb(features, tk, threshold=1024):

    pad_id = tk.pad_token_id
    assert pad_id is not None, "Tokenizer has no pad token"

    def pad_elems(ls, pad_id, maxlen):
        while len(ls) < maxlen:
            ls.append(pad_id)
        return ls

    maxlen = max([len(x["input_ids"]) for x in features])
    # avoid attention_type switching
    if maxlen < threshold:
        maxlen = threshold

    # dynamic padding
    input_ids = [pad_elems(x["input_ids"], pad_id, maxlen) for x in features]
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    # padding mask
    attention_mask = input_ids.clone()
    attention_mask[attention_mask != pad_id] = 1
    attention_mask[attention_mask == pad_id] = 0

    # tokenize answers
    tokenized_answers_int = tk.batch_encode_plus(
        [x["a1"] for x in features], add_special_tokens=False
    )["input_ids"]
    tokenized_answers_tensor = [torch.Tensor(x) for x in tokenized_answers_int]
    tokenized_answers = stack_with_padding(tokenized_answers_tensor, pad_id)

    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": torch.tensor(
            [x["labels"]["start_token"] for x in features],
            dtype=torch.long,  # cleanup by removing ["labels"]?
        ),
        "end_positions": torch.tensor(
            [x["labels"]["end_token"] for x in features],
            dtype=torch.long,  # cleanup by removing ["labels"]
        ),
        "pooler_label": torch.tensor(
            [CATEGORY_MAPPING[x["labels"]["category"][0]] for x in features]
        ),
        "gt_answers": tokenized_answers,
    }
    return output


def check_tokenizer(tokenizer):
    """make sure the tokenizer handles padding correctly"""
    assert tokenizer.decode([0]) == ""


def standardize_padding(tokens):
    """make sure the padding token is always 0 and not -100 as set by the huggingface trainer.
    inputs:
        Tokens: a 2d array of token ids
    outputs:
        Tokens: a 2d tensor of token ids with padding token 0
    """
    tokens = torch.tensor(tokens)
    tokens[tokens == -100] = 0
    return tokens


def find_sublist_in_list(sl, l):
    """Return the index of the first occurrence of the sublist `sl` in the list `l`"""
    result = -1
    subsequence_length = len(sl)
    for i in range(len(l)):
        if l[i : i + subsequence_length] == sl:
            result = i
            break
    return result

def sublist_is_in_list(sl, l):
    """Return whether the sublist `sl` is in the list `l`"""
    subsequence_length = len(sl)
    for i in range(len(l)):
        if l[i : i + subsequence_length] == sl:
            return True
    return False

def get_save_path(save_dir, downsample_pt_size, m1_arch, m2_arch, oracle_arch, template_id, step):
    save_path = os.path.join(
        save_dir,
        f"analysis_dataset_{'full' if downsample_pt_size is None else downsample_pt_size}_{m1_arch}_{m2_arch}_{oracle_arch}_{template_id}_{step}.json",
    )
    return save_path


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


def stack_with_padding(tensor_list, pad_id):
    max_len = max([t.shape[0] for t in tensor_list])
    padded_tensor_list = []
    for i in range(len(tensor_list)):
        container = torch.zeros(max_len, dtype=tensor_list[i].dtype) + pad_id
        container[: tensor_list[i].shape[0]] = tensor_list[i]
        padded_tensor_list.append(container)
    output = torch.stack(padded_tensor_list, dim=0)
    return output


def unpad_and_tokenize_single(tokenized_answer, tk):
    # unpad
    tokenized_answer = tokenized_answer[tokenized_answer != tk.pad_token_id]
    # tokenize
    tokenized_answer = tk.convert_ids_to_tokens(tokenized_answer)
    return tokenized_answer

def unpad_and_tokenize(tokenized_answer, tk):
    output = []
    for ta in tokenized_answer:
        output.append(unpad_and_tokenize_single(ta, tk))
    return output


def unstack_with_padding(tensor, lengths):
    return [t[:l] for t, l in zip(tensor, lengths)]


def collate_fn(features, tk, threshold=1024):
    pad_id = tk.pad_token_id

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
        [x["answer"] for x in features], add_special_tokens=False
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

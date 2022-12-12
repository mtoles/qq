# from transformers.utils import logging
import logging
import torch
import torch.nn as nn
from transformers import BigBirdForQuestionAnswering

from prepare_data import CATEGORY_MAPPING
from dataset_utils import *

INVERSE_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}


def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]


def _is_match_single(
    start_logits,
    end_logits,
    cls_logits,
    input_ids,
    start_idx_gt,
    end_idx_gt,
    cls_gt,
    tk,
):
    cls_out = cls_logits.argmax()
    cat_pred = INVERSE_CATEGORY_MAPPING[cls_out.item()]

    start_idx_pred, end_idx_pred = get_best_valid_start_end_idx(
        torch.Tensor(start_logits), torch.Tensor(end_logits), top_k=8, max_size=16
    )
    # Let's convert the input ids back to actual tokens
    answer_tokens_pred = input_ids[start_idx_pred : end_idx_pred + 1]

    # get ground truth answers
    category = INVERSE_CATEGORY_MAPPING[cls_gt]
    if category in ["yes", "no"]:
        answer_gt = set([category])
    else:
        answer_gt = expand_to_aliases(
            [tk.decode(input_ids[start_idx_gt : end_idx_gt + 1])],
            make_sub_answers=True,
        )

    # get predicted answers
    if cat_pred in ["yes", "no"]:
        model_output = cat_pred
    else:
        model_output = tk.decode(answer_tokens_pred)

    predictions = expand_to_aliases([model_output])

    # if there is a common element, it's a match
    match = len(list(answer_gt & predictions)) > 0
    return match


def is_match(
    start_logits,
    end_logits,
    cls_logits,
    input_ids,
    start_idx_gt,
    end_idx_gt,
    cls_gt,
    tk,
):
    matches = []
    for i in range(len(start_logits)):
        matches.append(
            _is_match_single(
                start_logits[i],
                end_logits[i],
                cls_logits[i],
                input_ids[i],
                start_idx_gt[i],
                end_idx_gt[i],
                cls_gt[i],
                tk,
            )
        )
    accuracy = sum(matches) / len(matches)
    return {"accuracy": accuracy}


def compute_metrics(tk, x):
    start_logits, end_logits, cls_logits, input_ids = x[0]
    cls_gt, start_gt, end_gt = x[1]

    accuracy = is_match(
        start_logits, end_logits, cls_logits, input_ids, start_gt, end_gt, cls_gt, tk
    )
    return accuracy


# Set logging to log level WARN
bb_logger = logging.getLogger("transformers.models.big_bird.modeling_big_bird")
bb_logger.setLevel("WARN")


def collate_fn(features, pad_id=0, threshold=1024):
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
    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": torch.tensor(
            [x["start_token"] for x in features], dtype=torch.long
        ),
        "end_positions": torch.tensor(
            [x["end_token"] for x in features], dtype=torch.long
        ),
        "pooler_label": torch.tensor([x["category"] for x in features]),
    }
    return output


class BigBirdForNaturalQuestions(BigBirdForQuestionAnswering):
    """BigBirdForQuestionAnswering with CLS Head over the top for predicting category"""

    def __init__(self, config):
        super().__init__(config, add_pooling_layer=True)
        self.cls = nn.Linear(config.hidden_size, 5)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        pooler_label=None,
    ):

        outputs = super().forward(input_ids, attention_mask=attention_mask)
        cls_out = self.cls(outputs.pooler_output)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            start_loss = loss_fct(outputs.start_logits, start_positions)
            end_loss = loss_fct(outputs.end_logits, end_positions)

            if pooler_label is not None:
                cls_loss = loss_fct(cls_out, pooler_label)
                loss = (start_loss + end_loss + cls_loss) / 3
            else:
                loss = (start_loss + end_loss) / 2

        return {
            "loss": loss,
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "cls_out": cls_out,
            "input_ids": input_ids,
        }

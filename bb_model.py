# from transformers.utils import logging
import logging
import torch
import torch.nn as nn
import numpy as np
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


def _get_metrics_single(
    start_logits,
    end_logits,
    cls_logits,
    input_ids,
    start_idx_gt,
    end_idx_gt,
    cls_gt,
    log_path,
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
        answer_gt = category
        answer_gt_expanded = set([category])
    else:
        # TODO: needs to take in actual ground truth text and compare rather than use indexing
        # start_idx_gt==-1 indicates answer does not exist in context.
        raise NotImplementedError  # needs implementing
        if start_idx_gt >= 0:
            answer_gt = tk.decode(input_ids[start_idx_gt : end_idx_gt + 1])
            answer_gt_expanded = expand_to_aliases(
                [answer_gt],
                make_sub_answers=True,
            )
        elif start_idx_gt == -1:
            answer_gt = None
        else:
            raise ValueError("start_idx_gt should be -1 or >=0")

    # get predicted answers
    if cat_pred in ["yes", "no"]:
        model_output = cat_pred
    else:
        model_output = tk.decode(answer_tokens_pred)

    predictions = expand_to_aliases([model_output])

    # if there is a common element, it's a match
    match = len(list(answer_gt_expanded & predictions)) > 0

    # f1
    tp = sum([word in answer_gt for word in model_output.split(" ")])
    fp = sum([word not in answer_gt for word in model_output.split(" ")])
    fn = sum([word not in model_output for word in answer_gt.split(" ")])
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    if log_path is not None:
        log_results(
            log_path,
            start_idx_pred,
            end_idx_pred,
            cls_logits,
            input_ids,
            start_idx_gt,
            end_idx_gt,
            cls_gt,
            match,
            f1,
            precision,
            recall,
            tk,
        )
    return match, f1, precision, recall


def log_results(
    log_path,
    start_idx_pred,
    end_idx_pred,
    cls_logits,
    input_ids,
    start_idx_gt,
    end_idx_gt,
    cls_gt,
    match,
    f1,
    precision,
    recall,
    tk,
):
    # ignore input_ids where the input_id value is -100
    input_ids = input_ids[input_ids != -100]
    context = tk.decode(input_ids).replace("[SEP]", "\n\n")
    log_entry = f"""{context}
answer: {tk.decode(input_ids[start_idx_gt : end_idx_gt + 1])}
prediction: {tk.decode(input_ids[start_idx_pred : end_idx_pred + 1])}
cls_gt: {cls_gt}
cls_pred: {cls_logits.argmax()}
match: {match}
f1: {str(f1)[:5]}
precision: {str(precision)[:5]}
recall: {str(recall)[:5]}
======================================
"""
    with open(log_path, "a") as f:
        f.write(log_entry)


def get_metrics(
    start_logits,
    end_logits,
    cls_logits,
    input_ids,
    start_idx_gt,
    end_idx_gt,
    cls_gt,
    log_path,
    tk,
):
    matches = []
    f1s = []
    precisions = []
    recalls = []
    for i in range(len(start_logits)):
        m, f, p, r = _get_metrics_single(
            start_logits[i],
            end_logits[i],
            cls_logits[i],
            input_ids[i],
            start_idx_gt[i],
            end_idx_gt[i],
            cls_gt[i],
            log_path,
            tk,
        )
        matches.append(m)
        f1s.append(f)
        precisions.append(p)
        recalls.append(r)

    accuracy = sum(matches) / len(matches)
    f1 = sum(f1s) / len(f1s)
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


# TODO: needs to be moved outside the model and made model agnostic
def compute_metrics(tk, log_path, x):
    start_logits, end_logits, cls_logits, input_ids = x[0]
    cls_gt, start_gt, end_gt = x[1]

    metrics = get_metrics(
        start_logits,
        end_logits,
        cls_logits,
        input_ids,
        start_gt,
        end_gt,
        cls_gt,
        log_path,
        tk,
    )
    return metrics


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
    }
    return output


class BigBirdForNaturalQuestions(BigBirdForQuestionAnswering):
    """BigBirdForQuestionAnswering with CLS Head over the top for predicting category"""

    def __init__(self, config, tk):
        super().__init__(config, add_pooling_layer=True)
        self.cls = nn.Linear(config.hidden_size, 5)
        self.tk = tk

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

        # Compute Loss

        loss = None
        # if start_positions is not None and end_positions is not None:
        assert (
            start_positions is not None and end_positions is not None
        ), "something bad happening"
        loss_fct = nn.CrossEntropyLoss()
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        if self.training:
            # only compute loss during training since validation will have
            # unanswerable questions, thus loss is not defined
            start_loss = loss_fct(outputs.start_logits, start_positions)
            end_loss = loss_fct(outputs.end_logits, end_positions)
            # cls_loss is still defined during validation, but it is not used
            if pooler_label is not None:
                cls_loss = loss_fct(cls_out, pooler_label)
                loss = (start_loss + end_loss + cls_loss) / 3
            else:
                raise ValueError("pooler_label is None")  # shouldn't happen?
                # loss = (start_loss + end_loss) / 2
        else:
            loss = torch.tensor([np.nan])

        # Get the Answer as a String
        pred_str = self.get_pred_str(
            input_ids, outputs.start_logits, outputs.end_logits
        )

        return {
            "loss": loss,
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "cls_out": cls_out,
            # "input_ids": input_ids,
            "input_ids": None,  # TODO: drop once verified unnecessary
            "pred_str": pred_str,
        }

    def get_pred_str(self, input_ids, start_logits, end_logits):
        pred_strs = []
        for i in range(len(start_logits)):

            start_idx_pred, end_idx_pred = get_best_valid_start_end_idx(
                torch.Tensor(start_logits[i]),
                torch.Tensor(end_logits[i]),
                top_k=8,
                max_size=16,
            )
            # Let's convert the input ids back to actual tokens
            answer_tokens_pred = input_ids[i][start_idx_pred : end_idx_pred + 1]
            pred_str = self.tk.decode(answer_tokens_pred)
            pred_strs.append(pred_str)
        return pred_str

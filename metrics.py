import torch
from utils import (
    INVERSE_CATEGORY_MAPPING,
    unpad_and_tokenize,
)
from dataset_utils import expand_to_aliases
from hotpot_evaluate_v1 import f1_score

def compute_metrics(x, tk, log_path):
    """cls predictions and ground truths are passed as integer categories.
    answers predictions are passed as tokenized sequences.
    however, answer ground truths are passed as start and end indices.
    This is the only way to compare span prediction models to generative models.
    """
    start_logits, end_logits, cls_pred, tokens_pred, input_ids = x[0]
    cls_gt, start_gt, end_gt, tokens_gt = x[1]

    str_pred = unpad_and_tokenize(tokens_pred, tk)
    str_gt = unpad_and_tokenize(tokens_gt, tk)

    metrics = get_metrics(
        str_pred,
        str_gt,
        cls_pred,
        cls_gt,
        input_ids,
        tk,
        log_path,
    )
    return metrics


def get_metrics(
    str_pred,
    str_gt,
    cls_pred,
    cls_gt,
    input_ids,
    tk,
    log_path,
):
    matches = []
    f1s = []
    precisions = []
    recalls = []
    for i in range(len(str_pred)):
        m, f, p, r = get_metrics_single(
            str_pred[i],
            str_gt[i],
            cls_pred[i],
            cls_gt[i],
            input_ids[i],
            tk,
            log_path,
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


def get_metrics_single(
    str_pred,
    str_gt,
    cls_pred,
    cls_gt,
    input_ids,
    tk,
    log_path,
):
    cat_pred = INVERSE_CATEGORY_MAPPING[cls_pred.item()]
    # get ground truth answers
    category = INVERSE_CATEGORY_MAPPING[cls_gt]
    if category in ["yes", "no"]:
        answer_gt = category
        answer_gt_expanded = set([category])
    # else:
    #     # TODO: needs to take in actual ground truth text and compare rather than use indexing
    #     # start_idx_gt==-1 indicates answer does not exist in context.
    #     if start_idx_gt >= 0:
    #         answer_gt = tk.decode(input_ids[start_idx_gt : end_idx_gt + 1])
    #         answer_gt_expanded = expand_to_aliases(
    #             [answer_gt],
    #             make_sub_answers=True,
    #         )
    #     elif start_idx_gt == -1:
    #         answer_gt = None
    #     else:
    #         raise ValueError("start_idx_gt should be -1 or >=0")

    # get predicted answers
    if cat_pred in ["yes", "no"]:
        str_pred = cat_pred
    # else:
    #     model_output = unpad_and_tokenize_single(tokenized_answer, tk)
    #     # model_output = tk.decode(answer_tokens_pred)
    # answer_gt_expanded = expand_to_aliases([str_gt], make_sub_answers=True)

    # predictions = expand_to_aliases([str_pred])

    # # if there is a common element, it's a match
    # match = len(list(answer_gt_expanded & predictions)) > 0

    # # f1
    # tp = sum([word in str_gt for word in str_pred.split(" ")])
    # fp = sum([word not in str_gt for word in str_pred.split(" ")])
    # fn = sum([word not in str_pred for word in str_gt.split(" ")])
    # try:
    #     precision = tp / (tp + fp)
    # except ZeroDivisionError:
    #     precision = 0
    # try:
    #     recall = tp / (tp + fn)
    # except ZeroDivisionError:
    #     recall = 0
    # try:
    #     f1 = 2 * (precision * recall) / (precision + recall)
    # except ZeroDivisionError:
        # f1 = 0
    match = 0
    f1, precision, recall = f1_score(str_pred, str_gt)
    if log_path is not None:
        log_results(
            str_pred,
            str_gt,
            cls_pred,
            cls_gt,
            match,
            f1,
            precision,
            recall,
            input_ids,
            tk,
            log_path,
        )
    return match, f1, precision, recall


def log_results( 
    str_pred,
    str_gt,
    cls_pred,
    cls_gt,
    match,
    f1,
    precision,
    recall,
    input_ids,
    tk,
    log_path,
):
    context = tk.decode(input_ids)
    log_entry = f"""{context}
answer: {str_pred}
prediction: {str_gt}
cls_pred: {cls_pred}
cls_gt: {cls_gt}
match: {match}
f1: {str(f1)[:5]}
precision: {str(precision)[:5]}
recall: {str(recall)[:5]}
======================================
"""
    with open(log_path, "a") as f:
        f.write(log_entry)


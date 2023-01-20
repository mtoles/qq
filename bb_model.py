# from transformers.utils import logging
import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import BigBirdForQuestionAnswering


from dataset_utils import *
from utils import INVERSE_CATEGORY_MAPPING, CATEGORY_MAPPING, stack_with_padding


def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]


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
        pred_tokens = self.get_tokens_pred(
            input_ids, outputs.start_logits, outputs.end_logits
        )

        return {
            "loss": loss,
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "cls_out": cls_out,
            # "input_ids": input_ids,
            # "input_ids": None,  # TODO: drop once verified unnecessary
            "pred_str": pred_tokens,
        }

    def get_tokens_pred(self, input_ids, start_logits, end_logits):
        # TODO: vectorize
        answer_tokens_preds = []
        for i in range(len(start_logits)):

            start_idx_pred, end_idx_pred = get_best_valid_start_end_idx(
                torch.Tensor(start_logits[i]),
                torch.Tensor(end_logits[i]),
                top_k=8,
                max_size=16,
            )
            # Let's convert the input ids back to actual tokens
            answer_tokens_pred = input_ids[i][start_idx_pred : end_idx_pred + 1]
            answer_tokens_preds.append(answer_tokens_pred)
        preds_as_tensor = stack_with_padding(answer_tokens_preds)
        return preds_as_tensor

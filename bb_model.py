# from transformers.utils import logging
import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import BigBirdForQuestionAnswering


from dataset_utils import *
from utils import (
    INVERSE_CATEGORY_MAPPING,
    CATEGORY_MAPPING,
    stack_with_padding,
    collate_fn_bb,
)


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
        gt_answers=None,
    ):
        question_lengths = torch.zeros(
            input_ids.shape[1]
        ).cuda()  # allow super to look for answers in the question. I think this is not actually done in the original BB implementation (or vasadevgupta's), but it might just be an artifact from repurposing the natural questions code. Either way I think it's fine and I don't want to rewrite the whole thing.
        outputs = super().forward(
            input_ids, attention_mask=attention_mask, question_lengths=question_lengths
        )

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

        # Get the Answer as Tokens
        tokens_pred = self.get_tokens_pred(
            input_ids, outputs.start_logits, outputs.end_logits
        )

        # Get the Predicted Category
        cls_pred = cls_out.argmax(axis=1)
        if loss > 1e5:
            print(f"ALERT: loss is big!: {loss}")
        return {
            "loss": loss,
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "cls_pred": cls_pred,
            "pred_str": tokens_pred,
            "input_ids": input_ids,  # keep here so we can log them later
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
        preds_as_tensor = stack_with_padding(answer_tokens_preds, self.tk.pad_token_id)
        return preds_as_tensor

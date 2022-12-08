# from transformers.utils import logging
import logging
import torch
import torch.nn as nn
from transformers import BigBirdForQuestionAnswering



# Set logging to log level WARN
bb_logger = logging.getLogger('transformers.models.big_bird.modeling_big_bird')
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
        }

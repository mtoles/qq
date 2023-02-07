# A wrapper class for primary models
# Define custom primary models here

from utils import BB_MODEL_ID, GPT_NEO_MODEL_ID
from bb_model import BigBirdForNaturalQuestions
from utils import collate_fn
from metrics import compute_metrics
from transformers import (
    BigBirdTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    # Trainer,
    PreTrainedModel,
)
from transformers import Trainer
from prepare_data import prepare_inputs_hp

import numpy as np
import torch


class Primary_Model(PreTrainedModel):
    def __init__(
        self,
        model_path=None,
        eval_batch_size=2,
        raw_val_dataset=None,
        prepped_val_dataset=None,
    ):
        self.model_path = model_path
        self.eval_batch_size = eval_batch_size
        self.raw_val_dataset = raw_val_dataset
        self.prepped_val_dataset = prepped_val_dataset
        self.collate_fn = lambda x: collate_fn(x, self.tk)
        # don't compute the loss
        self.args = TrainingArguments(
            output_dir="main/outputs/",
            do_train=False,
            do_eval=True,
            per_gpu_eval_batch_size=eval_batch_size,
            group_by_length=True,
            disable_tqdm=False,
            remove_unused_columns=False,
            label_names=[
                "pooler_label",
                "start_positions",
                "end_positions",
                "gt_answers",
            ],
        )

    def __call__(self, **inputs):
        return self.forward(**inputs)

    def forward(self, **inputs):
        """Forward pass must return     
    start_logits, end_logits, cls_pred, tokens_pred, input_ids = x[0]
    cls_gt, start_gt, end_gt, tokens_gt = x[1]
Check `compute_metrics` function for actual requirements"""
        return self.model(**inputs)

    def prepare_data(self, masking_scheme):
        self.trainer = Trainer(
            model=self,
            args=self.args,
            data_collator=self.collate_fn,
            eval_dataset=self.prepped_val_dataset,
            compute_metrics=lambda x: compute_metrics(x, self.tk, None),
            tokenizer=self.tk,
        )


class BigBird_PM(Primary_Model):
    def __init__(
        self,
        model_path=None,
        eval_batch_size=2,
        raw_val_dataset=None,
        prepped_val_dataset=None,
    ):
        super(PreTrainedModel, self).__init__()
        self.tk = BigBirdTokenizer.from_pretrained(BB_MODEL_ID)
        self.model = BigBirdForNaturalQuestions.from_pretrained(model_path, self.tk)
        super(BigBird_PM, self).__init__(
            model_path=model_path,
            eval_batch_size=eval_batch_size,
            raw_val_dataset=raw_val_dataset,
            prepped_val_dataset=prepped_val_dataset,
        )

    def prepare_data(self, masking_scheme):
        self.prepped_val_dataset = self.raw_val_dataset.map(
            lambda x: prepare_inputs_hp(
                x,
                tk=self.tk,
                max_length=self.model.bert.embeddings.position_embeddings.weight.shape[
                    0
                ],
                masking_scheme=masking_scheme,
            )
        )
        super().prepare_data(self.prepped_val_dataset)


class GPTNeoX_PM(Primary_Model):
    def __init__(
        self,
        eval_batch_size=2,
        raw_val_dataset=None,
        prepped_val_dataset=None,
    ):
        super(PreTrainedModel, self).__init__()
        self.tk = AutoTokenizer.from_pretrained(GPT_NEO_MODEL_ID)
        self.tk.pad_token_id = 1  # specific to gpt-neo-x
        self.model = AutoModelForCausalLM.from_pretrained(GPT_NEO_MODEL_ID)
        super(GPTNeoX_PM, self).__init__(
            model_path=None,
            eval_batch_size=eval_batch_size,
            raw_val_dataset=raw_val_dataset,
            prepped_val_dataset=prepped_val_dataset,
        )

    def forward(self, **inputs):
        """    start_logits, end_logits, cls_pred, tokens_pred, input_ids = x[0]
    cls_gt, start_gt, end_gt, tokens_gt = x[1]"""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        assert attention_mask.sum() != 0, "attention_mask must not be all zeros"
        generation = self.model.generate(input_ids, attention_mask=attention_mask)
        loss, outputs = torch.tensor(np.nan), generation

        start_logits=None
        end_logits=None
        cls_pred=None

        return (loss, outputs)

    def prepare_data(self, masking_scheme):
        masking_str = f"fc_{masking_scheme}"
        # replace [SEP] with :
        def _replace_sep(x):
            assert (
                len(x[masking_str].split("[SEP]")) == 2
            ), "masking_str must contain exactly one [SEP]"
            x[masking_str] = x[masking_str].replace("[SEP]", "\n\n")
            return x

        # Prepare the dataset
        self.prepped_val_dataset = self.raw_val_dataset.map(lambda x: _replace_sep(x))
        self.prepped_val_dataset = self.prepped_val_dataset.map(
            lambda x: prepare_inputs_hp(
                x,
                tk=self.tk,
                max_length=2048,
                masking_scheme=masking_scheme,
            )
        )

        super().prepare_data(self.prepped_val_dataset)

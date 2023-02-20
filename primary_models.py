# A wrapper class for primary models
# Define custom primary models here

from utils import *
from bb_model import BigBirdForNaturalQuestions
from utils import collate_fn_bb
from metrics import compute_metrics_bb, get_metrics
from transformers import (
    BigBirdTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
)
from prepare_data import prepare_inputs_hp, prepend_question
from tqdm import tqdm

import torch


class Primary_Model:
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

    def __call__(self, **inputs):
        return self.forward(**inputs)

    def forward(self, **inputs):
        """Forward pass must return
            start_logits, end_logits, cls_pred, tokens_pred, input_ids = x[0]
            cls_gt, start_gt, end_gt, tokens_gt = x[1]
        Check `compute_metrics` function for actual requirements"""
        return self.model(**inputs)

    def prepare_data(self, masking_scheme):
        print("You should subclass this method")
        pass

    def evaluate(self):
        print("You should subclass this method")
        pass


class BigBird_PM(Primary_Model):
    """This is an example of how to wrap a model based on the hugging face trainer."""

    def __init__(
        self,
        model_path=None,
        eval_batch_size=2,
        raw_val_dataset=None,
        prepped_val_dataset=None,
    ):
        # super(PreTrainedModel, self).__init__()
        self.collate_fn = lambda x: collate_fn_bb(x, self.tk)
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
        self.tk = BigBirdTokenizer.from_pretrained(BB_MODEL_ID)
        self.model = BigBirdForNaturalQuestions.from_pretrained(model_path, self.tk)
        super(BigBird_PM, self).__init__(
            model_path=model_path,
            eval_batch_size=eval_batch_size,
            raw_val_dataset=raw_val_dataset,
            prepped_val_dataset=prepped_val_dataset,
        )

    def prepare_data(self, masking_scheme, save_input_ids=False):
        """
        Method for preparing the validation dataset for evaluation.

        Args:
            masking_scheme (str):
                The masking scheme to use. Usually "randomsentence"
            save_as_input_ids (bool, optional):
                Whether to save the dataset as input_ids or not.
                Some functions assume the dataset has only one "input_ids" column, so be
        """
        self.prepped_val_dataset = self.raw_val_dataset.map(
            lambda x: prepend_question(x, masking_scheme, self.tk.sep_token)
        )
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
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            data_collator=self.collate_fn,
            eval_dataset=self.prepped_val_dataset,
            compute_metrics=lambda x: compute_metrics_bb(x, self.tk, None),
            tokenizer=self.tk,
        )

    def evaluate(self, masking_scheme):
        # Prep the dataset again so that the input_ids are generated from the masking_scheme column
        self.prepare_data(masking_scheme, save_input_ids=True)
        evaluation = self.trainer.evaluate()
        # Drop the input_ids so they aren't used by mistake later
        self.prepped_val_dataset = self.prepped_val_dataset.remove_columns("input_ids")
        # remove the "eval_" prefix from each dictionary key
        evaluation = {k[5:]: v for k, v in evaluation.items() if k.startswith("eval_")}
        return evaluation


class T5_PM(Primary_Model):
    def __init__(
        self,
        eval_batch_size=2,
        raw_val_dataset=None,
        prepped_val_dataset=None,
        model_name=None,
    ):
        model_name = f"google/flan-{model_name}"
        self.tk = AutoTokenizer.from_pretrained(model_name, cache_dir="./.model_cache")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir="./.model_cache"
        ).cuda()
        super(T5_PM, self).__init__(
            model_path=None,
            eval_batch_size=eval_batch_size,
            raw_val_dataset=raw_val_dataset,
            prepped_val_dataset=prepped_val_dataset,
        )

    def forward(self, **inputs):
        """Perform forward pass on a single example. Not sure what happens with padding if you pass multiple examples."""
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).cuda()
        generation = self.model.generate(
            input_ids, max_new_tokens=10, pad_token_id=self.tk.pad_token_id
        )[:, 1:-1]
        generation_str = self.tk.decode(generation[0])
        return generation

    def prepare_data(self, masking_scheme):
        masking_str = f"fc_{masking_scheme}"
        # replace [SEP] with :
        def _replace_sep(x):
            assert (
                len(x[masking_str].split("[SEP]")) == 2
            ), "masking_str must contain exactly one [SEP]"
            x[masking_str] = x[masking_str].replace(
                "[SEP]", "\n\n"
            )  # TODO: fix. this is getting stripped out by the split/join
            return x

        def _add_prompt(x):
            x[masking_str] = x[masking_str] + "\n\nAnswer in as few words as possible: "
            return x

        # Prepare the dataset
        # self.prepped_val_dataset = self.raw_val_dataset.map(lambda x: _replace_sep(x))
        self.prepped_val_dataset = self.raw_val_dataset.map(
            lambda x: prepend_question(x, masking_scheme, "\n\n")
        )
        self.prepped_val_dataset = self.prepped_val_dataset.map(
            lambda x: _add_prompt(x)
        )
        self.prepped_val_dataset = self.prepped_val_dataset.map(
            lambda x: prepare_inputs_hp(
                x,
                tk=self.tk,
                max_length=2048,
                masking_scheme=masking_scheme,
            )
        )

    def evaluate(self, masking_scheme):
        masking_str = f"fc_{masking_scheme}"
        with torch.no_grad():
            str_pred = []
            str_gt = []
            cls_pred = []
            cls_gt = []
            input_ids = []

            for i, x in enumerate(tqdm(self.prepped_val_dataset)):
                input_tokens = self.tk(self.prepped_val_dataset[i][masking_str])[
                    "input_ids"
                ]
                generation = self.forward(input_ids=input_tokens)
                generation_str = self.tk.decode(generation[0])
                str_pred.append(
                    self.tk.batch_decode(generation, skip_special_tokens=True)[0]
                )
                str_gt.append(x["answer"])
                cls_pred.append(None)
                cls_gt.append(None)
                input_ids.append(input_tokens)

            metrics = get_metrics(
                str_pred,
                str_gt,
                cls_pred,
                cls_gt,
                input_ids,
                self.tk,
                None,
            )
        return metrics

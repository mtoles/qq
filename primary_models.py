# A wrapper class for primary models
# Define custom primary models here
# Currently only t5 models are supported.
# Bigbird needs to be updated following code changes elsewhere.

from utils import *
from bb_model import BigBirdForNaturalQuestions
from utils import collate_fn_bb
from metrics import compute_metrics_bb, get_metrics
from transformers import (
    BigBirdTokenizer,
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from prepare_data import (
    prepare_inputs_hp,
    prepend_question,
    append_a2,
)
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import numpy as np
from datasets.utils.logging import disable_progress_bar, enable_progress_bar


class Primary_Model:
    def __init__(
        self,
        model_path=None,
        batch_size=2,
    ):
        self.model_path = model_path
        self.batch_size = batch_size

    def __call__(self, **inputs):
        return self.forward(**inputs)

    def forward(self, **inputs):
        """Forward pass must return
            start_logits, end_logits, cls_pred, tokens_pred, input_ids = x[0]
            cls_gt, start_gt, end_gt, tokens_gt = x[1]
        Check `compute_metrics` function for actual requirements"""
        return self.model(**inputs)

    def prepare_data(self, masking_scheme, a2_col, raw_ds):
        ds = raw_ds
        # TODO: add column for masking_scheme
        new_col_name = f"prepped_{masking_scheme}_{str(a2_col)}"
        if new_col_name not in ds.column_names:
            ds = ds.add_column(
                column=raw_ds[f"fc_{masking_scheme}"],
                name=new_col_name,
            )

        return ds

    def evaluate(self):
        raise NotImplementedError("You should subclass this method")


class BigBird_PM(Primary_Model):
    """This is an example of how to wrap a model based on the hugging face trainer."""

    def __init__(
        self,
        model_path=None,
        batch_size=2,
    ):
        self.collate_fn = lambda x: collate_fn_bb(x, self.tk)
        # don't compute the loss
        self.args = TrainingArguments(
            output_dir="main/outputs/",
            do_train=False,
            do_eval=True,
            per_gpu_eval_batch_size=batch_size,
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
        if model_path is not None:
            self.model = BigBirdForNaturalQuestions.from_pretrained(
                model_path, self.tk
            ).cuda()
        else:
            print("You're loading an untrained model. Are you sure?")
            self.model = BigBirdForNaturalQuestions.from_pretrained(
                BB_MODEL_ID, self.tk
            ).cuda()
        super(BigBird_PM, self).__init__(
            model_path=model_path,
            batch_size=batch_size,
        )
        self.max_length = self.model.bert.embeddings.position_embeddings.weight.shape[0]

    def prepare_data(self, masking_scheme, raw_val_dataset, a2_col):
        """
        Method for preparing the validation dataset for evaluation.

        Args:
            masking_scheme (str):
                The masking scheme to use. Usually "randomsentence"
        """
        prepped_val_dataset = super(BigBird_PM, self).prepare_data(
            masking_scheme=masking_scheme,
            a2_col=a2_col,
            raw_ds=raw_val_dataset,
        )
        sep_str = "\n\n" if self.tk.sep_token is None else self.tk.sep_token
        prepped_val_dataset = prepped_val_dataset.map(
            lambda x: prepend_question(x, masking_scheme, sep_str),
            load_from_cache_file=False,
        )
        if a2_col is not None:
            prepped_val_dataset = prepped_val_dataset.map(
                lambda x: append_a2(x, masking_scheme, sep_str),
                load_from_cache_file=False,
            )
        prepped_val_dataset = prepped_val_dataset.map(
            lambda x: prepare_inputs_hp(
                x,
                tk=self.tk,
                max_length=self.max_length,
                masking_scheme=masking_scheme,
            ),
            load_from_cache_file=False,
        )
        return prepped_val_dataset

    def evaluate(self, masking_scheme, a2_col, ds):
        # Prep the dataset again so that the input_ids are generated from the masking_scheme column
        prepped_dataset = self.prepare_data(
            masking_scheme=masking_scheme,
            raw_val_dataset=ds,
            a2_col=a2_col,
        )
        trainer = Trainer(
            model=self.model,
            args=self.args,
            data_collator=self.collate_fn,
            eval_dataset=prepped_dataset,
            compute_metrics=lambda x: compute_metrics_bb(x, self.tk, None),
            tokenizer=self.tk,
        )

        evaluation = trainer.evaluate()
        # drop the "eval_" prefix
        evaluation = {k[5:]: v for k, v in evaluation.items()}
        return evaluation


class T5_PM(Primary_Model):
    def __init__(
        self,
        batch_size=2,
        model_name=None,
    ):
        model_name = f"google/flan-{model_name}"
        self.tk = AutoTokenizer.from_pretrained(model_name, cache_dir="./.model_cache")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir="./.model_cache"
        ).cuda()
        super(T5_PM, self).__init__(
            model_path=None,
            batch_size=batch_size,
        )

    def forward(self, **inputs):
        """Perform forward pass on a single example. Not sure what happens with padding if you pass multiple examples."""
        input_ids = inputs["input_ids"]
        generation = self.model.generate(
            input_ids, max_new_tokens=10, pad_token_id=self.tk.pad_token_id
        )[:, 1:-1]
        generation_str = self.tk.decode(generation[0])
        return generation

    def prepare_data(self, masking_scheme, raw_val_dataset, a2_col):
        # Call the parent class's prepare_data method to get the prepped_val_dataset
        prepped_val_dataset = super(T5_PM, self).prepare_data(
            masking_scheme=masking_scheme,
            a2_col=a2_col,
            raw_ds=raw_val_dataset,
        )

        def _add_prompt(x, masking_scheme):
            x[f"prepped_{masking_scheme}_{str(a2_col)}"] = (
                x[f"prepped_{masking_scheme}_{str(a2_col)}"]
                + "\n\nAnswer in as few words as possible: "
            )
            return x

        prepped_val_dataset = prepped_val_dataset.map(
            lambda x: prepend_question(x, masking_scheme, a2_col, "\n\n"),
            load_from_cache_file=False,
        )

        if a2_col is not None:
            prepped_val_dataset = prepped_val_dataset.map(
                lambda x: append_a2(x, masking_scheme, a2_col, "\n\n"),
                load_from_cache_file=False,
            )
        prepped_val_dataset = prepped_val_dataset.map(
            lambda x: _add_prompt(x, masking_scheme),
            load_from_cache_file=False,
        )
        return prepped_val_dataset

    def evaluate(
        self, masking_scheme, ds, a2_col, max_adversarial_examples=None, threshold=None, display=True
    ):
        """
        Args:
            masking_scheme (str): The masking scheme to use. Usually "bfsentence"
            ds (Dataset): The dataset to evaluate on. Must have a column named "prepped_{masking_scheme}_{str(a2_col)}"
            a2_col (str): The name of the column containing the answer to the question. If None, then no answer is given.
            max_adversarial_examples (int): The maximum number of adversarial examples to evaluate. If None, then as many as possible will be evaluated.
        """
        with torch.no_grad():
            assert (max_adversarial_examples and threshold) or (
                not max_adversarial_examples and not threshold
            ), "Must specify both max_adversarial_examples and threshold (adversarial mode) or neither (normal mode)"
            adversarial_mode = max_adversarial_examples is not None
            # adversarial_mode = False
            # If in adversarial mode, shuffle the dataset to remove biases
            if adversarial_mode:
                ds = ds.shuffle()
            masking_str = f"prepped_{masking_scheme}_{str(a2_col)}"
            # ds = self.prepare_data(masking_scheme, ds, a2_col)
            disable_progress_bar()
            ds = self.prepare_data(masking_scheme, ds, a2_col)

            # Data used for computing aggregate metrics
            str_preds = []
            str_gts = []
            cls_preds = []
            cls_gts = []
            input_idss = []

            # Data recorded into the dataset under ['m1_{masking_scheme}_gen', 'm1_{masking_scheme}_f1']
            gen_strs = []  # generated strings
            f1s = []  # f1 scores
            ems = []  # eexact match (binary)

            num_batches = len(ds) // self.batch_size
            if len(ds) % self.batch_size != 0:
                num_batches += 1

            num_adversarial_examples = 0
            batch_ids = tqdm(range(num_batches)) if display else range(num_batches)
            for batch_idx in batch_ids:
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(ds))
                batch = ds.select(range(start_idx, end_idx))
                input_tokens = self.tk(
                    batch[masking_str], return_tensors="pt", padding=True
                )["input_ids"].cuda()
                generation = self.forward(input_ids=input_tokens)
                str_preds_batch = self.tk.batch_decode(
                    generation, skip_special_tokens=True
                )
                str_preds.extend(str_preds_batch)
                str_gts.extend([x["a1"] for x in batch])
                cls_preds.extend([None] * len(batch))
                cls_gts.extend([None] * len(batch))
                input_idss.extend(input_tokens)
                batch_metrics = get_metrics(
                    str_preds_batch,
                    [x["a1"] for x in batch],
                    [None] * len(batch),
                    [None] * len(batch),
                    input_tokens,
                    self.tk,
                    None,
                )
                gen_strs.extend(str_preds_batch)
                f1s.extend(batch_metrics["f1"])
                ems.extend(batch_metrics["em"])

                # Break the loop if we are in adversarial generation mode
                # and have generated enough adversarial examples.
                if adversarial_mode:
                    original_f1s = batch[f"m1_supporting_{str(a2_col)}_f1"]
                    adversarial_f1s = batch_metrics["f1"]
                    was_damaged_batch = [
                        x - y < threshold for x, y in zip(original_f1s, adversarial_f1s)
                    ]
                    num_new_adversarial_examples = sum(was_damaged_batch)
                    if num_new_adversarial_examples >= max_adversarial_examples:
                        # If we end early, truncate the dataset to only the data that actually got processed
                        ds = ds.select(range(len(gen_strs)))
                        break

            # If in adversarial mode, only keep the first max_adversarial_examples examples where

            ds = ds.add_column(f"m1_{masking_scheme}_{str(a2_col)}_gen", gen_strs)
            ds = ds.add_column(f"m1_{masking_scheme}_{str(a2_col)}_f1", f1s)
            ds = ds.add_column(f"m1_{masking_scheme}_{str(a2_col)}_em", ems)

            # Reduce to only the first max_adversarial_examples examples where the model was damaged
            if adversarial_mode:
                ds = ds.filter(
                    lambda x: x[f"m1_supporting_{str(a2_col)}_f1"]
                    - x[f"m1_{masking_scheme}_{str(a2_col)}_f1"]
                    > threshold,
                    num_proc=1,
                ).shuffle()
                ds = ds.select(range(min(max_adversarial_examples, len(ds))))
            # Get aggregate metrics
            # Note that aggregate metrics are not very meaningful if in adversarial mode
            # Since the dataset is filtered to only include examples where the model was damaged
            agg_f1 = np.mean(ds[f"m1_{masking_scheme}_{str(a2_col)}_f1"])
            agg_em = np.mean(ds[f"m1_{masking_scheme}_{str(a2_col)}_em"])

            metrics = {
                "f1": agg_f1,
                "em": agg_em,
            }
            # assert (not max_adversarial_examples) or (
            #     len(ds) <= max_adversarial_examples
            # )
        enable_progress_bar()
        return ds, metrics


def get_m1(m1_path, m1_arch, batch_size):
    # Unit Tests
    assert m1_arch in [
        "bigbird",
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-xl",
        "t5-xxl",
    ]
    # Load primary model
    if m1_arch == "bigbird":
        m1 = BigBird_PM(m1_path, batch_size=batch_size)
    elif m1_arch.startswith("t5"):
        m1 = T5_PM(
            batch_size=batch_size,
            model_name=m1_arch,
        )
    else:
        raise NotImplementedError
    return m1

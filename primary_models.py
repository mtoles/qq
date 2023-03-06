# A wrapper class for primary models
# Define custom primary models here

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

import torch


class Primary_Model:
    def __init__(
        self,
        model_path=None,
        eval_batch_size=2,
    ):
        self.model_path = model_path
        self.eval_batch_size = eval_batch_size

    def __call__(self, **inputs):
        return self.forward(**inputs)

    def forward(self, **inputs):
        """Forward pass must return
            start_logits, end_logits, cls_pred, tokens_pred, input_ids = x[0]
            cls_gt, start_gt, end_gt, tokens_gt = x[1]
        Check `compute_metrics` function for actual requirements"""
        return self.model(**inputs)

    def prepare_data(self, masking_scheme, a2_col, raw_val_dataset):
        prepped_val_dataset = raw_val_dataset
        # TODO: add column for masking_scheme
        prepped_val_dataset = prepped_val_dataset.add_column(
            column=raw_val_dataset[f"fc_{masking_scheme}"],
            name=f"prepped_{masking_scheme}_{str(a2_col)}",
        )

        return prepped_val_dataset

    def evaluate(self):
        raise NotImplementedError("You should subclass this method")


class BigBird_PM(Primary_Model):
    """This is an example of how to wrap a model based on the hugging face trainer."""

    def __init__(
        self,
        model_path=None,
        eval_batch_size=2,
    ):
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
            eval_batch_size=eval_batch_size,
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
            raw_val_dataset=raw_val_dataset,
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
        eval_batch_size=2,
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
        )

    def forward(self, **inputs):
        """Perform forward pass on a single example. Not sure what happens with padding if you pass multiple examples."""
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).cuda()
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
            raw_val_dataset=raw_val_dataset,
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
        prepped_val_dataset = prepped_val_dataset.map(
            lambda x: prepare_inputs_hp(
                x,
                tk=self.tk,
                max_length=2048,
                masking_scheme=masking_scheme,
            ),
            load_from_cache_file=False,
        )
        return prepped_val_dataset

    def evaluate(self, masking_scheme, ds, a2_col):
        masking_str = f"prepped_{masking_scheme}_{str(a2_col)}"
        ds = self.prepare_data(masking_scheme, ds, a2_col)

        with torch.no_grad():
            # Data used for computing aggregate metrics
            str_preds = []
            str_gts = []
            cls_preds = []
            cls_gts = []
            input_idss = []

            # Data recorded into the dataset under ['m1_{masking_scheme}_gen', 'm1_{masking_scheme}_f1']
            gen_strs = []  # generated strings
            f1s = []  # f1 scores

            for i, x in enumerate(tqdm(ds)):
                input_tokens = self.tk(ds[i][masking_str])["input_ids"]
                generation = self.forward(input_ids=input_tokens)
                str_preds.append(
                    self.tk.batch_decode(generation, skip_special_tokens=True)[0]
                )
                generation_str = str_preds[-1]
                str_gts.append(x["a1"])
                cls_preds.append(None)
                cls_gts.append(None)
                input_idss.append(input_tokens)
                single_metrics = get_metrics(
                    str_preds[-1:],
                    str_gts[-1:],
                    cls_preds[-1:],
                    cls_gts[-1:],
                    input_idss[-1:],
                    self.tk,
                    None,
                )
                gen_strs.append(generation_str)
                f1s.append(single_metrics["f1"])

            ds = ds.add_column(f"m1_{masking_scheme}_{str(a2_col)}_gen", gen_strs)
            ds = ds.add_column(f"m1_{masking_scheme}_{str(a2_col)}_f1", f1s)

            # Get aggregate metrics
            metrics = get_metrics(
                str_preds,
                str_gts,
                cls_preds,
                cls_gts,
                input_idss,
                self.tk,
                None,
            )
        return ds, metrics

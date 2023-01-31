# A wrapper class for primary models
# Define custom primary models here

from utils import BB_MODEL_ID
from bb_model import BigBirdForNaturalQuestions
from utils import collate_fn
from metrics import compute_metrics
from transformers import BigBirdTokenizer
from prepare_data import prepare_inputs_hp
from transformers import TrainingArguments, Trainer


class Primary_Model:
    def __init__(
        self,
        model_path=None,
        eval_batch_size=32,
        raw_val_dataset=None,
        prepped_val_dataset=None,
    ):
        self.model_path = model_path
        self.eval_batch_size = eval_batch_size
        self.raw_val_dataset = raw_val_dataset
        self.prepped_val_dataset = prepped_val_dataset
        self.collate_fn = lambda x: collate_fn(x, self.tk)
        self.args = TrainingArguments(
            output_dir=None,
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

    def __call__(self, batch):
        self.forward(batch)

    def forward(self, batch):
        self.model(batch)

    def prepare_data(self, masking_scheme):
        self.trainer = Trainer(
            model=self.model,
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
        eval_batch_size=32,
        raw_val_dataset=None,
        prepped_val_dataset=None,
    ):
        self.tk = BigBirdTokenizer.from_pretrained(BB_MODEL_ID)
        if model_path is None:
            self.model = BigBirdForNaturalQuestions.from_pretrained(
                BB_MODEL_ID, self.tk
            )
        else:
            self.model = BigBirdForNaturalQuestions.from_pretrained(model_path, self.tk)
        super(BigBird_PM, self).__init__(
            model_path=model_path,
            eval_batch_size=eval_batch_size,
            raw_val_dataset=raw_val_dataset,
            prepped_val_dataset=prepped_val_dataset,
        )

    def __call__(self, example):
        return self.model(example)

    def forward(self, example):
        return self.model(example)

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
        super().prepare_data(self.raw_val_dataset)

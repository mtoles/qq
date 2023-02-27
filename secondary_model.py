from transformers import PreTrainedModel, GPT2Tokenizer, GPT2Model


class Dummy_Secondary_Model:
    def __init__(
        self,
        eval_batch_size=2,
        raw_val_dataset=None,
        prepped_val_dataset=None,
        model_name="gpt2",
    ):
        self.model_name = "dummy"
        self.tk = GPT2Tokenizer.from_pretrained(model_name, cache_dir="./.model_cache")
        self.model = GPT2Model.from_pretrained(
            model_name, cache_dir="./.model_cache"
        ).cuda()
        self.eval_batch_size = eval_batch_size
        self.raw_val_dataset = raw_val_dataset
        self.prepped_val_dataset = prepped_val_dataset

    def prepare_data(self, masking_scheme):
        pass

    def forward(self, example, question_col, context_col):
        # return "What is six times seven?"
        return example["masked_sentence"]

    def process(self, ds, primary_question_col, context_col):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'secondary_question'."""

        def _add_secondary_question(example):
            example["secondary_question"] = self.forward(
                example, primary_question_col, context_col
            )
            return example

        ds = ds.add_column(
            name="secondary_question", column=[""] * len(ds)
        )
        ds = ds.map(
            lambda x: _add_secondary_question(x),
            load_from_cache_file=False,
        )
        return ds

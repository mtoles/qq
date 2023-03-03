from transformers import PreTrainedModel, GPT2Tokenizer, GPT2Model


class Dummy_Secondary_Model:
    def __init__(
        self,
        eval_batch_size=2,
        model_name="gpt2",
    ):
        self.model_name = "dummy"

    def prepare_data(self, masking_scheme):
        pass

    def forward(self, example, question_col, context_col):
        return "What is six times seven?"
        # return example["masked_sentence"]

    def process(self, ds, q1_col, context_col):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""

        def _add_q2(example):
            example["q2"] = self.forward(example, q1_col, context_col)
            return example

        ds = ds.add_column(name="q2", column=[""] * len(ds))
        ds = ds.map(
            lambda x: _add_q2(x),
            load_from_cache_file=False,
        )
        return ds


class Repeater_Secondary_Model:
    def __init__(
        self,
        eval_batch_size=2,
        model_name="gpt2",
    ):
        self.model_name = "dummy"
        # self.tk = GPT2Tokenizer.from_pretrained(model_name, cache_dir="./.model_cache")
        # self.model = GPT2Model.from_pretrained(
        #     model_name, cache_dir="./.model_cache"
        # ).cuda()

    def prepare_data(self, masking_scheme):
        pass

    def forward(self, example, question_col, context_col):
        # Always return the original question q1
        return example[question_col]

    def process(self, ds, q1_col, context_col):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""

        def _add_q2(example):
            example["q2"] = self.forward(example, q1_col, context_col)
            return example

        ds = ds.add_column(name="q2", column=[""] * len(ds))
        ds = ds.map(
            lambda x: _add_q2(x),
            load_from_cache_file=False,
        )
        return ds

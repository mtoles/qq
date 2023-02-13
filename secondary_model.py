from transformers import PreTrainedModel, GPT2Tokenizer, GPT2Model


class Dummy_Secondary_Model():
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

    def forward(self, input_ids):
        return "What is six times seven?"

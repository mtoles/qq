from transformers import PreTrainedModel


class Dummy_secondary_model(PreTrainedModel):
    def forward(self, input_ids):
        return "What is six times seven?"

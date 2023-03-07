from transformers import PreTrainedModel, GPT2Tokenizer, GPT2Model
import openai
import configparser

# Set up the API once for all models
config = configparser.ConfigParser()
config.read("config.ini")
openai.api_key = config.get("API_KEYS", "openai_api_key")

# Abstract class for secondary models
class Secondary_Model:
    def __init__(
        self,
    ):
        self.model_name = "dummy"

    def prepare_data(self, masking_scheme):
        pass

    def forward(self, example, question_col, context_col):
        return "What is six times seven?"

    def process(self, ds, q1_col, masking_scheme):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""

        def _add_q2(example):
            example[f"q2_{masking_scheme}"] = self.forward(example, q1_col, f"fc_{masking_scheme}")
            return example

        ds = ds.add_column(name=f"q2_{masking_scheme}", column=[""] * len(ds))
        ds = ds.map(
            lambda x: _add_q2(x),
            load_from_cache_file=False,
        )
        return ds


class Repeater_Secondary_Model(Secondary_Model):
    def __init__(
        self,
    ):
        self.model_name = "repeater"

    def prepare_data(self, masking_scheme):
        pass

    def forward(self, example, question_col, context_col):
        # Always return the original question q1
        return example[question_col]


class OpenAI_Secondary_Model(Secondary_Model):
    def __init__(
        self,
    ):
        self.model_name = "chatGPT"
        self.model = "gpt-3.5-turbo"
        # call the openai api with a test prompt

    def prepare_data(self, masking_scheme):
        pass

    def forward(self, example, question_col, context_col):
        # Always return the original question q1
        q1 = example[question_col]
        context = example[context_col]
        prompt = f"Ask another question that would help you answer the following question:\n\n{context}\n\n{q1}"

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        q2 = response["choices"][0]["message"]["content"].strip()
        return q2



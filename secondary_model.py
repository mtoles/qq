from transformers import PreTrainedModel, GPT2Tokenizer, GPT2Model
import pandas as pd
import numpy as np
import openai
import configparser
import os
import h5py

from tqdm import tqdm

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
            example[f"q2_{masking_scheme}"] = self.forward(
                example, q1_col, f"fc_{masking_scheme}"
            )
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

    def forward(self, example, question_col, context_col):
        # Always return the original question q1
        return example[question_col]


class OpenAI_Secondary_Model(Secondary_Model):
    def __init__(self, cache_path):
        self.model_name = "chatGPT"
        self.model = "gpt-3.5-turbo"
        self.cache_path = cache_path
        self.oai_model_id = (
            None  # the current openai model id. set on the first api call
        )

        if not os.path.exists(self.cache_path):
            self.cache_df = pd.DataFrame(columns=["response"])
            self.cache_df.to_csv(self.cache_path)
            # Create the cache file
        else:
            self.cache_df = pd.read_csv(self.cache_path, index_col=0)

    def prepare_data(self, masking_scheme):
        pass

    def forward(self, example, question_col, context_col):
        def call_oai_api(prompt):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
            except Exception as e:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                    )
                except Exception as e:
                    try:
                        response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=[
                                {"role": "user", "content": prompt},
                            ],
                        )
                    except Exception as e:
                        print(f"OpenAI API call failed: {e}")
                        return "OpenAI API call failed"
            q2 = response["choices"][0]["message"]["content"].strip()
            self.oai_model_id = response.model
            idx = f"{self.oai_model_id} {prompt}"
            # Cache the response in ram
            new_line = pd.DataFrame(columns=["response"], index=[idx], data=[q2])
            self.cache_df = pd.concat([self.cache_df, new_line])
            new_line.to_csv(self.cache_path, mode="a", header=False, escapechar="🦆")
            return q2

        # If we don't yet know the current OpenAi model id, get it
        q1 = example[question_col]
        context = example[context_col]
        prompt = f"Ask another question that would help you answer the following question:\n\n{context}\n\n{q1}"
        if self.oai_model_id is not None:
            idx = f"{self.oai_model_id} {prompt}"
        else:
            return call_oai_api(prompt)
        # Check if the response is cached
        if idx in self.cache_df.index and self.oai_model_id is not None:
            return self.cache_df.loc[idx, "response"]
        else:
            return call_oai_api(prompt)

    def process(self, ds, q1_col, masking_scheme):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""

        def _add_q2(example):
            example[f"q2_{masking_scheme}"] = self.forward(
                example, q1_col, f"fc_{masking_scheme}"
            )
            return example

        ds = ds.add_column(name=f"q2_{masking_scheme}", column=[""] * len(ds))
        # for i in tqdm(range(len(ds))):
        #     ds[i] = _add_q2(ds[i])
        ds = ds.map(
            lambda x: _add_q2(x),
            load_from_cache_file=False,
        )
        return ds


class Gt_Secondary_Model(Secondary_Model):
    def __init__(
        self,
    ):
        self.model_name = "groundtruth"
        self.gt_q2_path = "q2_gt_dataset.csv"
        self.df = pd.read_csv(self.gt_q2_path)

    def forward(self, example, question_col, context_col):
        # Always return the original question q1
        if example["id"] in self.df["id"].values:
            gt_q2 = self.df[self.df["id"] == example["id"]]["gt_q2"].values[0]
            if gt_q2 is not np.nan:
                return gt_q2
            else:
                return f"id {example['id']} has empty gt_q2 in {self.gt_q2_path}"
        else:
            return f"id {example['id']} not found in {self.gt_q2_path}"

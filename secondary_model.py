import math
import torch
from transformers import (
    PreTrainedModel,
    GPT2Tokenizer,
    GPT2Model,
    LlamaForCausalLM,
    BitsAndBytesConfig,
)
import pandas as pd
import numpy as np
import openai
import configparser
import os
import time
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import SFTTrainer
from copy import deepcopy

# Set up the API once for all models
config = configparser.ConfigParser()
config.read("config.ini")
openai.api_key = config.get("API_KEYS", "openai_api_key")


EXAMPLES = [
    {
        "context": 'House of Anubis: House of Anubis is a mystery television series developed for Nickelodeon based on the Dutch-Belgian television series "Het Huis Anubis".',
        "q1": 'The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?',
        "q2": 'In what year did "Het Huis Anubis" air?',
    },
    {
        "context": "TOberoi family: The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
        "q1": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "q2": "What city holds the head office of The Oberoi Group?",
    },
    {
        "context": "Zilpo Road: The nine mile byway starts south of Morehead, Kentucky and can be accessed by U.S. Highway 60. Arkansas Highway 113: The route runs 29.48 mi from Arkansas Highway 10 to Morrilton.",
        "q1": "What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?",
        "q2": "what U.S. Highway is also known as Midland Trail?",
    },
    {
        "context": 'My Finale: "My Finale" is the hour-long season finale for season eight of the American sitcom "Scrubs". Human Error (House): "Human Error" is the twenty-fourth episode and season finale of the third season of "House" and the seventieth episode overall.',
        "q1": 'Human Error" is the season finale of the third season of a tv show that aired on what network?',
        "q2": 'What network aired the season finale of the third season of "House"?',
    },
    {
        "context": 'Annette Bening: Annette Carol Bening (born May 29, 1958) is an American actress. She is a four-time Academy Award nominee; for "The Grifters" (1990), "American Beauty" (1999), "Being Julia" (2004) and "The Kids Are All Right" (2010). In 2006, she received a star on the Hollywood Walk of Fame. The Great Outdoors (film): The Great Outdoors is a 1988 American comedy film directed by Howard Deutch, and written and produced by John Hughes. John Lithgow: John Arthur Lithgow ( ; born October 19 , 1945) is an American actor, musician, singer, comedian, voice actor, and author.',
        "q1": "The 1988 American comedy film, The Great Outdoors, starred a four-time Academy Award nominee, who received a star on the Hollywood Walk of Fame in what year?",
        "q2": "Who starred in the 1988 American comedy film, The Great Outdoors?",
    },
]

in_context_examples = ""
for i, ex in enumerate(EXAMPLES):
    in_context_examples += (
        f"Context:\n{ex['context']}\nQuestion 1:\n{ex['q1']}\nQuestion 2:\n{ex['q2']}\n"
    )


# Abstract class for secondary models
class Secondary_Model:
    def __init__(
        self,
        prompt_id="p1",
    ):
        self.model_name = "dummy"
        self.prompt_id = prompt_id  # the id of the prompt to use for this model
        self.prompt_dict = {
            "p1": "Ask another question that would help you answer the following question:\n\n{context}\n\n{q1}",
            "p2": "Some information is missing from this context. Ask a simpler question that would help you answer it.\n\nContext:\n\n{context}\n\nMain Question:\n\n{q1}\n\nSimpler question:",
            "p3": "What question can you ask to help you answer the final question?\n\n{context}\n\n{q1}\n\nYou can ask:",
            "p4": "".join(
                [
                    "Ask another question that would help you answer the previous question:\n\n",
                    in_context_examples,
                    "\n\nContext:\n{context}\nQuestion 1:\n{q1}\nQuestion 2:\n",
                ]
            ),
            "p5": "".join(
                [
                    "Some information is missing from each context. Ask a simpler question that would help you answer it.\n\n",
                    in_context_examples,
                    "\n\nContext:\n{context}\nQuestion 1:\n{q1}\nQuestion 2:\n",
                ]
            ),
            "p6": "".join(
                [
                    "What question can you ask to help you answer the final question?\n\n",
                    in_context_examples,
                    "\n\nContext:\n{context}\nQuestion 1:\n{q1}\nQuestion 2:\n",
                ]
            ),
            "p2_jeopardy": "Some information is missing from this context. Ask a simpler question that would help you answer it.\n\nContext:\n\n{context}\n\nMain Question:\n\n{q1}\n\nSimpler question:\n\nMake sure your question could be answered by this passage:\n\n{masked_sentence}",
        }
        self.template = self.prompt_dict[self.prompt_id]

    def prepare_data(self):
        pass

    def forward(self, example, question_col, context_col):
        return "What is six times seven?"

    def process(self, ds, q1_col):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""

        def _add_q2(example):
            example[f"q2"] = self.forward(example, q1_col, "fc_masked")
            return example

        ds = ds.add_column(name=f"q2", column=[""] * len(ds))
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


class Alpaca_Secondary_Model(Secondary_Model):
    def __init__(
        self,
        model_name,
        model_path,
        max_length=2048,
        prompt_id="p3",
        device="cuda",
        precision="bf16",
        quantization_config=None,
        tokenizer_path=None,
        eval_batch_size=1,
    ):
        super(Alpaca_Secondary_Model, self).__init__(prompt_id)
        self.model_name = model_name
        self.device = device
        self.eval_batch_size = eval_batch_size

        # if loading the alepaca model you need to manually pass in the original tokenizer
        # because i forgot to save the tokenzier with the model
        if tokenizer_path is None:
            self.tokenizer_path = model_path
        else:
            self.tokenizer_path = tokenizer_path

        if precision == "bf16":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
        elif precision == "int8":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=True,
                quantization_config=quantization_config,
            )
        # low-memory fine tuning option
        elif precision == "bnb_4":
            print("loading alpaca in bnb_4 bit")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,  # Same quantization config as before
                device_map="auto",
                trust_remote_code=True,
            )

            # self.model = PeftModel.from_pretrained(
            #     normal_model, "models/alpaca-jeopardy-1-epoch"
            # )

            # # trainer = SFTTrainer(
            # #     model=self.model,
            # #     # peft_config=peft_config,
            # #     dataset_text_field="training_example",
            # #     max_seq_length=max_seq_length,
            # #     tokenizer=tokenizer,
            # #     args=training_arguments,
            # #     packing=packing,
            # # )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        if str(self.model.device) == "cpu":
            self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.max_length = max_length
        self.alpaca_template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\nResponse:"
        # self.alpaca_template_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    def fit_template(self, q1, context):
        instruction_templates = {
            "p1": "Ask another question that would help you answer the following question:",
            "p2": "Some information is missing from this context. Ask a simpler question that would help you answer it.",
            "p3": "What question can you ask to help you answer the final question?",
            "p4": "Ask another question that would help you answer the following question:",
            "p5": "Some information is missing from this context. Ask a simpler question that would help you answer it.",
            "p6": "What question can you ask to help you answer the final question?",
        }
        input_templates = {
            "p1": "Question:\n{q1}\nContext:\n{context}",
            "p2": "Context:\n{context}\nMain Question:\n{q1}\nSimpler question:",
            "p3": "Context:\n{context}\nQuestion:\n{q1}\nYou can ask:",
            "p4": "Question:\n{q1}\nContext:\n{context}",
            "p5": "Context:\n{context}\nMain Question:\n{q1}\nSimpler question:",
            "p6": "Context:\n{context}\nQuestion:\n{q1}\nYou can ask:",
        }

        if self.prompt_id in ["p1", "p2", "p3", "p4", "p5", "p6"]:
            instruction = instruction_templates[self.prompt_id]
            inpt = input_templates[self.prompt_id].format(context=context, q1=q1)
            prompt = self.alpaca_template.format(instruction=instruction, input=inpt)
            if self.prompt_id in ["p4", "p5", "p6"]:
                inputs = []
                for i, ex in enumerate(EXAMPLES):
                    inpt = input_templates[self.prompt_id].format(
                        context=ex["context"], q1=ex["q1"]
                    )
                    # inputs.append(self.alpaca_template.format(instruction=instruction, input=input) + "\n" + ex['q2'])
                    inputs.append(inpt + "\nResponse:\n" + ex["q2"])
                inputs.append(
                    input_templates[self.prompt_id].format(context=context, q1=q1)
                )
                inputs = "\n\n".join(inputs)
                prompt = self.alpaca_template.format(
                    instruction=instruction, input=inputs
                )

        else:
            raise Exception("No such prompt")

        return prompt

    def forward(self, example, question_col, context_col):
        q1 = example[question_col]
        context = example[context_col]
        prompt = self.fit_template(q1, context)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {
            k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
        }
        with torch.no_grad():
            eos_token_id = self.tokenizer.eos_token_id
            outputs = self.model.generate(
                **inputs, max_length=self.max_length, eos_token_id=eos_token_id
            )
            q2 = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False
            )

        return q2

    def process(self, ds, q1_col):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""
        # ds = ds.add_column(name=f"q2", column=[""] * len(ds))
        num_batches = math.ceil(len(ds) / self.eval_batch_size)
        q2s = []

        for i in tqdm(range(num_batches)):
            start = i * self.eval_batch_size
            end = min((i + 1) * self.eval_batch_size, len(ds))
            batch = ds.select(list(range(start, end)))
            prompts = []
            for j in range(len(batch)):
                q1 = batch[j][q1_col]
                context = batch[j]["fc_masked"]
                prompts.append(self.fit_template(q1, context))
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {
                k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
            }
            with torch.no_grad():
                eos_token_id = self.tokenizer.eos_token_id
                outputs = self.model.generate(
                    **inputs, max_length=self.max_length, eos_token_id=eos_token_id
                )
                q2s.extend(
                    self.tokenizer.batch_decode(
                        outputs[:, len(inputs["input_ids"][0]) :],
                        skip_special_tokens=True,
                    )
                )
        # for i in range(len(ds)):
        #     ds[i]["q2"] = q2s[i]

        ds = ds.add_column(name=f"q2", column=q2s)
        return ds


class Alpaca_Secondary_Model_Jeopardy_Lookup(Secondary_Model):
    """ "
    Secondary model that uses precomputed secondary questions seeded with the answer
    Used to evaluate the viability of Jeopardy-style training
    """

    def __init__(
        self, precomputed_jeopardy_path: str, model_name="alexpaca_precomputed"
    ):
        self.model_name = model_name
        self.precomputed_jeopardy_path = precomputed_jeopardy_path
        self.df = pd.read_json(self.precomputed_jeopardy_path, lines=True)
        # set  fc_masked as the index
        self.df.set_index("q1", inplace=True)

    def forward(self, example, question_col, answer_col):
        key = example["q1"]
        return self.df.loc[key, "jeopardy_q"]


class OpenAI_Secondary_Model(Secondary_Model):
    def __init__(self, cache_path, model_name, prompt_id="p1"):
        super(OpenAI_Secondary_Model, self).__init__(prompt_id)
        # self.model_name = "chatGPT"
        # self.model_name = model_name
        if model_name == "gpt-3.5-turbo":
            self.model_name = "gpt-3.5-turbo-1106"
        elif model_name == "gpt-4":
            self.model_name = "gpt-4-0314"
        # self.model = "gpt-3.5-turbo"
        self.cache_path = cache_path
        self.oai_model_id = (
            None  # the current openai model id. set on the first api call
        )
        if self.cache_path is None:
            self.cache_df = pd.DataFrame(columns=["response"])
        else:
            if not os.path.exists(self.cache_path):
                self.cache_df = pd.DataFrame(columns=["response"])
                self.cache_df.to_csv(self.cache_path)
                # Create the cache file
            else:
                self.cache_df = pd.read_csv(self.cache_path, index_col=0)

    def prepare_data(self, masking_scheme):
        pass

    def forward(self, example, question_col, context_col):
        idx = None

        def call_oai_api(prompt):
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                    )
                    break
                except Exception as e:
                    print(e)
                    print("Retrying...")
                    # pause a second
                    time.sleep(1)
                    continue

            q2 = response["choices"][0]["message"]["content"].strip()
            self.oai_model_id = response.model
            idx = f"{self.oai_model_id} {prompt}"
            # Cache the response in ram
            if idx in self.cache_df.index:
                self.cache_df.loc[idx, "response"] = q2
                self.cache_df.to_csv(self.cache_path)
            else:
                new_line = pd.DataFrame(columns=["response"], index=[idx], data=[q2])
                self.cache_df = pd.concat([self.cache_df, new_line])
                new_line.to_csv(
                    self.cache_path, mode="a", header=False, escapechar="🦆"
                )
            return q2

        q1 = example[question_col]
        context = example[context_col]
        # prompt = f"Ask another question that would help you answer the following question:\n\n{context}\n\n{q1}"
        prompt = self.template.format(context=context, q1=q1)

        # if self.oai_model_id is not None:
        #     idx = f"{self.oai_model_id} {prompt}"
        # else:
        #     # Check if the response is cached
        #     # If not, call the API to get the model id
        #     output = call_oai_api(prompt)
        # if idx in self.cache_df.index and self.oai_model_id is not None:
        #     output = self.cache_df.loc[idx, "response"]
        # else:
        #     output = call_oai_api(prompt)
        output = call_oai_api(prompt)
        # assert type(output) == str, f"OpenAI API call failed: {output}"
        return output

    def process(self, ds, q1_col):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""

        def _add_q2(example):
            example[f"q2"] = self.forward(example, q1_col, f"fc_masked")
            return example

        ds = ds.add_column(name=f"q2", column=[""] * len(ds))
        # for i in tqdm(range(len(ds))):
        #     _add_q2(ds[i])  # debugging
        ds = ds.map(
            lambda x: _add_q2(x),
            load_from_cache_file=False,
        )
        return ds


class OpenAI_Jeopardy_Secondary_Model(Secondary_Model):
    def __init__(self, cache_path, model_name):
        super(OpenAI_Jeopardy_Secondary_Model, self).__init__("p2_jeopardy")
        # self.model_name = "chatGPT"
        self.model_name = model_name
        # self.model = "gpt-3.5-turbo"
        self.cache_path = cache_path
        self.oai_model_id = (
            None  # the current openai model id. set on the first api call
        )
        if self.cache_path is None:
            self.cache_df = pd.DataFrame(columns=["response"])
        else:
            if not os.path.exists(self.cache_path):
                self.cache_df = pd.DataFrame(columns=["response"])
                self.cache_df.to_csv(self.cache_path)
                # Create the cache file
            else:
                self.cache_df = pd.read_csv(self.cache_path, index_col=0)

    def prepare_data(self, masking_scheme):
        pass

    def forward(self, example, question_col, context_col, masked_sentence_col):
        idx = None

        def call_oai_api(prompt):
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                    )
                    break
                except Exception as e:
                    print(e)
                    print("Retrying...")
                    # pause a second
                    time.sleep(1)
                    continue

            q2 = response["choices"][0]["message"]["content"].strip()
            self.oai_model_id = response.model
            idx = f"{self.oai_model_id} {prompt}"
            # Cache the response in ram
            if idx in self.cache_df.index:
                self.cache_df.loc[idx, "response"] = q2
                self.cache_df.to_csv(self.cache_path)
            else:
                new_line = pd.DataFrame(columns=["response"], index=[idx], data=[q2])
                self.cache_df = pd.concat([self.cache_df, new_line])
                new_line.to_csv(
                    self.cache_path, mode="a", header=False, escapechar="🦆"
                )
            return q2

        q1 = example[question_col]
        context = example[context_col]
        masked_sentence = example[masked_sentence_col]
        # prompt = f"Ask another question that would help you answer the following question:\n\n{context}\n\n{q1}"
        prompt = self.template.format(
            context=context, q1=q1, masked_sentence=masked_sentence
        )

        # if self.oai_model_id is not None:
        #     idx = f"{self.oai_model_id} {prompt}"
        # else:
        #     # Check if the response is cached
        #     # If not, call the API to get the model id
        #     output = call_oai_api(prompt)
        # if idx in self.cache_df.index and self.oai_model_id is not None:
        #     output = self.cache_df.loc[idx, "response"]
        # else:
        #     output = call_oai_api(prompt)
        output = call_oai_api(prompt)
        # assert type(output) == str, f"OpenAI API call failed: {output}"
        return output

    def process(self, ds, q1_col):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""

        def _add_q2(example):
            example[f"q2"] = self.forward(example, q1_col, f"fc_masked")
            return example

        ds = ds.add_column(name=f"q2", column=[""] * len(ds))
        # for i in tqdm(range(len(ds))):
        #     _add_q2(ds[i])  # debugging
        ds = ds.map(
            lambda x: _add_q2(x),
            load_from_cache_file=False,
        )
        return ds


class Gt_Secondary_Model(Secondary_Model):
    def __init__(self, gt_df):
        self.model_name = "groundtruth"
        self.gt_df = gt_df
        # self.gt_q2_path = "q2_gt_dataset.csv"
        # self.df = pd.read_csv(self.gt_q2_path)

    def forward(self, example, question_col, context_col):
        # Always return the original question q1
        id = example["id"].split("_")[0]
        if id in self.gt_df["id"].values:
            gt_q2 = self.gt_df[self.gt_df["id"] == id]["q2_gt"].values[0]
            if gt_q2 is not np.nan:
                return gt_q2
            else:
                raise NotImplementedError(
                    f"id {example['id']} has empty gt_q2 in self.gt_df"
                )
        else:
            raise NotImplementedError(f"id {example['id']} not found in self.gt_df")


class Llama3_Secondary_Model(Secondary_Model):
    def __init__(
        self,
        model_path,
        prompt_id,
        model_name="llama3",
        max_length=2048,
        device="cuda",
        precision="bf16",
        # quantization_config=None,
        # tokenizer_path=None,
        eval_batch_size=1,
    ):
        super(Llama3_Secondary_Model, self).__init__(prompt_id)
        self.model_name = model_name
        self.device = device
        self.eval_batch_size = eval_batch_size

        # if loading the alepaca model you need to manually pass in the original tokenizer
        # because i forgot to save the tokenzier with the model
        # if tokenizer_path is None:
        #     self.tokenizer_path = model_path
        # else:
        #     self.tokenizer_path = tokenizer_path

        # if precision == "bf16":
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16
        )

        # self.model = PeftModel.from_pretrained(
        #     normal_model, "models/alpaca-jeopardy-1-epoch"
        # )

        # # trainer = SFTTrainer(
        # #     model=self.model,
        # #     # peft_config=peft_config,
        # #     dataset_text_field="training_example",
        # #     max_seq_length=max_seq_length,
        # #     tokenizer=tokenizer,
        # #     args=training_arguments,
        # #     packing=packing,
        # # )

        # else:
        #     self.model = AutoModelForCausalLM.from_pretrained(model_path)
        if str(self.model.device) == "cpu":
            self.model.to(self.device)

        # self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.max_length = max_length
        # self.llama3_template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        self.messages = [
            {
                "role": "system",
                "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Only write the response, nothing else.",
            },
            {
                "role": "user",
                "content": "Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
            },
        ]
        # self.alpaca_template_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    def fit_template(self, q1, context):
        instruction_templates = {
            "p1": "Ask another question that would help you answer the following question:",
            "p2": "Some information is missing from this context. Ask a simpler question that would help you answer it.",
            "p3": "What question can you ask to help you answer the final question?",
            "p4": "Ask another question that would help you answer the following question:",
            "p5": "Some information is missing from this context. Ask a simpler question that would help you answer it.",
            "p6": "What question can you ask to help you answer the final question?",
        }
        input_templates = {
            "p1": "Question:\n{q1}\nContext:\n{context}",
            "p2": "Context:\n{context}\nMain Question:\n{q1}\nSimpler question:",
            "p3": "Context:\n{context}\nQuestion:\n{q1}\nYou can ask:",
            "p4": "Question:\n{q1}\nContext:\n{context}",
            "p5": "Context:\n{context}\nMain Question:\n{q1}\nSimpler question:",
            "p6": "Context:\n{context}\nQuestion:\n{q1}\nYou can ask:",
        }

        if self.prompt_id in ["p1", "p2", "p3", "p4", "p5", "p6"]:
            instruction = instruction_templates[self.prompt_id]
            inpt = input_templates[self.prompt_id].format(context=context, q1=q1)
            # prompt = self.llama3_template.format(instruction=instruction, input=inpt)
            if self.prompt_id in ["p4", "p5", "p6"]:
                inputs = []
                for i, ex in enumerate(EXAMPLES):
                    inpt = input_templates[self.prompt_id].format(
                        context=ex["context"], q1=ex["q1"]
                    )
                    # inputs.append(self.alpaca_template.format(instruction=instruction, input=input) + "\n" + ex['q2'])
                    inputs.append(inpt + "\nResponse:\n" + ex["q2"])
                inputs.append(
                    input_templates[self.prompt_id].format(context=context, q1=q1)
                )
                in_context_examples = "\n\n".join(inputs)
                # prompt = self.llama3_template.format(
                #     instruction=instruction, input=inputs
                # )
                inpt += in_context_examples
            prompt = deepcopy(self.messages)
            prompt[1]["content"] = prompt[1]["content"].format(
                instruction=instruction, input=inpt
            )

        else:
            raise Exception("No such prompt")
        prompt = self.tokenizer.decode(
            self.tokenizer.apply_chat_template(prompt)
        )  # dumb but i don't want to rewrite stuff
        return prompt

    def forward(self, example, question_col, context_col):
        q1 = example[question_col]
        context = example[context_col]
        prompt = self.fit_template(q1, context)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {
            k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
        }
        with torch.no_grad():
            eos_token_id = self.tokenizer.eos_token_id
            outputs = self.model.generate(
                **inputs, max_length=self.max_length, eos_token_id=eos_token_id
            )
            q2 = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False
            )

        return q2

    def process(self, ds, q1_col):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""
        # ds = ds.add_column(name=f"q2", column=[""] * len(ds))
        num_batches = math.ceil(len(ds) / self.eval_batch_size)
        q2s = []

        for i in tqdm(range(num_batches)):
            start = i * self.eval_batch_size
            end = min((i + 1) * self.eval_batch_size, len(ds))
            batch = ds.select(list(range(start, end)))
            prompts = []
            for j in range(len(batch)):
                q1 = batch[j][q1_col]
                context = batch[j]["fc_masked"]
                prompts.append(self.fit_template(q1, context))
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {
                k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
            }
            with torch.no_grad():
                eos_token_id = self.tokenizer.eos_token_id
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                )
                clean_outputs = []
                decoded = self.tokenizer.batch_decode(
                    outputs[:, len(inputs["input_ids"][0]) :],
                    skip_special_tokens=True,
                )
                for output in decoded:
                    clean_output = output.split("assistant\n\n")[1].strip()
                    q2s.append(clean_output)
        # for i in range(len(ds)):
        #     ds[i]["q2"] = q2s[i]

        ds = ds.add_column(name=f"q2", column=q2s)
        return ds


class Llama3_FT_Secondary_Model(Secondary_Model):
    def __init__(
        self,
        model_path,
        model_name="llama3",
        max_length=2048,
        device="cuda",
        precision="bf16",
        # quantization_config=None,
        # tokenizer_path=None,
        eval_batch_size=1,
    ):
        self.model_name = model_name
        self.device = device
        self.eval_batch_size = eval_batch_size

        # if loading the alepaca model you need to manually pass in the original tokenizer
        # because i forgot to save the tokenzier with the model
        # if tokenizer_path is None:
        #     self.tokenizer_path = model_path
        # else:
        #     self.tokenizer_path = tokenizer_path

        # if precision == "bf16":
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )

        # else:
        #     self.model = AutoModelForCausalLM.from_pretrained(model_path)
        if str(self.model.device) == "cpu":
            self.model.to(self.device)

        # self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.max_length = max_length
        # self.llama3_template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"

        # self.alpaca_template_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    def fit_template(self, q1, context):
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Ask another question that would help you answer the following question:\n\nQuestion:\n{q1}\nContext:\n{context}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ).format(context=context, q1=q1)
        return prompt

    def forward(self, example, question_col, context_col):
        q1 = example[question_col]
        context = example[context_col]
        prompt = self.fit_template(q1, context)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {
            k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
        }
        with torch.no_grad():
            eos_token_id = self.tokenizer.eos_token_id
            outputs = self.model.generate(
                **inputs, max_length=self.max_length, eos_token_id=eos_token_id
            )
            q2 = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False
            )

        return q2

    def process(self, ds, q1_col):
        """Ask a secondary question about each primary question. Returns a new dataset with the secondary question added as a column called 'q2'."""
        # ds = ds.add_column(name=f"q2", column=[""] * len(ds))
        num_batches = math.ceil(len(ds) / self.eval_batch_size)
        q2s = []

        for i in tqdm(range(num_batches)):
            start = i * self.eval_batch_size
            end = min((i + 1) * self.eval_batch_size, len(ds))
            batch = ds.select(list(range(start, end)))
            prompts = []
            for j in range(len(batch)):
                q1 = batch[j][q1_col]
                context = batch[j]["fc_masked"]
                prompts.append(self.fit_template(q1, context))
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {
                k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
            }
            with torch.no_grad():
                eos_token_id = self.tokenizer.eos_token_id
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                )
                clean_outputs = []
                # print(self.tokenizer.batch_decode(outputs))
                decoded = self.tokenizer.batch_decode(
                    outputs[:, len(inputs["input_ids"][0]) :],
                    skip_special_tokens=True,
                )
                # for output in decoded:
                #     print(output)
                #     clean_output = output.split(
                #         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                #     )[1].strip()
                q2s.extend(decoded)
        # for i in range(len(ds)):
        #     ds[i]["q2"] = q2s[i]

        ds = ds.add_column(name=f"q2", column=q2s)
        return ds

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import alpaca.utils as utils
from torch.utils.data import Dataset
from transformers import Trainer
from datetime import datetime
import wandb
import os

from secondary_model import format_instruction_llama3


######## W&B Setup ########
os.environ["WANDB_PROJECT"] = "qq"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"  # false/end/checkpoint
os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")  # save W&B logs locally
os.environ["WANDB_CACHE_DIR"] = os.path.join(
    os.getcwd(), "wandb/cache"
)  # save W&B cache locally
os.environ["WANDB_CONFIG_DIR"] = os.path.join(
    os.getcwd(), "wandb/config"
)  # save W&B config locally

from main import main as analyze

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"
# DEFAULT_PAD_TOKEN = "<pad>"
# PROMPT_DICT = {
#     # "prompt_input": (
#     #     "Below is an instruction that describes a task, paired with an input that provides further context. "
#     #     "Write a response that appropriately completes the request.\n\n"
#     #     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     # ),
#     # "prompt_no_input": (
#     #     "Below is an instruction that describes a task. "
#     #     "Write a response that appropriately completes the request.\n\n"
#     #     "### Instruction:\n{instruction}\n\n### Response:"
#     # ),
#     "prompt_no_input": (  # llama3
#         "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
#         "{instruction}\n"
#         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
#     ),
# }


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct"
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    round_cutoff: int = field(
        default=None,
        metadata={"help": "Drop data from the dataset generated on rounds on or after round_cutoff."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    examples: int = field(
        default=1000, metadata={"help": "Number of examples to train on."}
    )
    estring: str = field(
        default="", metadata={"help": "identifier string for model and eval outputs"}
    )
    save_on_each_node: bool = field(
        default=False, metadata={"help": "Save model on each node."}
    )
    do_eval: bool = field(
        default=True, metadata={"help": "Run evaluation after training."}
    )
    


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            # max_length=tokenizer.model_max_length,
            max_length=275,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        formatting_fn: callable,
        round_cutoff: int,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)
        if round_cutoff is not None:
            list_data_dict = [
                example for example in list_data_dict if example["round"] <= round_cutoff
            ]

        logging.warning("Formatting inputs...")
        # prompt_input, prompt_no_input = (
        #     PROMPT_DICT["prompt_input"],
        #     PROMPT_DICT["prompt_no_input"],
        # )
        # prompt_no_input = PROMPT_DICT["prompt_no_input"]

        # sources0 = [
        #     prompt_no_input.format_map(example)
        #     for example in list_data_dict
        # ]

        q1s = [example["q1"] for example in list_data_dict]
        contexts = [example["fc_masked"] for example in list_data_dict]
        sources = [formatting_fn(q1, context) for q1, context in zip(q1s, contexts)]

        print("sources: ")
        print(sources[0])
        # targets = [f"{example['output']}" for example in list_data_dict]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, formatting_fn
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, formatting_fn=formatting_fn, round_cutoff=data_args.round_cutoff
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    now = datetime.now().strftime("%m_%d-%H:%M:%S")
    run_name = "alpaca-jeopardy"

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.max_steps = training_args.examples // (
        training_args.per_device_train_batch_size or 1
    )
    print(f"training on {training_args.examples} examples")
    training_args.report_to = ["wandb"]
    wandb.init(project="qq", name=f"{run_name}-{now}", config=training_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    # assert None not in (
    #     tokenizer.pad_token,
    #     tokenizer.eos_token,
    #     tokenizer.bos_token,
    #     tokenizer.unk_token,
    # )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # temporarily load llama3 to get the formatting fn
    def format_for_llama3(q1, context):
        # inpt, instruction = get_input_and_instruction(q1, context, "p3")
        # chat_message = Llama3_FT_Secondary_Model.fit_template(inpt)
        # chat_message = format_instruction(q1, context, "p3")
        chat_message = format_instruction_llama3(q1, context, "p3")
        return chat_message

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, formatting_fn=format_for_llama3
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    torch.cuda.synchronize()
    trainer.save_state()
    cutoff_str = f"_cuttoff_{data_args.round_cutoff}" if data_args.round_cutoff is not None else ""
    output_path = f"{training_args.output_dir}/{training_args.estring}/{training_args.examples}{cutoff_str}_{now}"
    trainer.save_model(output_dir=output_path)
    print(f"saving model to {output_path}")

    print("evaluating model")

    # make only the first gpu visible for eval
    torch.cuda.synchronize()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    except KeyError:
        pass
    
    if training_args.do_eval:
        analyze_df = analyze(
            split="validation",
            m1_arch="t5-base",
            m2_arch="llama3",
            template_id="p3",
            # alexpaca_precomputed_data_path=output_path,
            alexpaca_precomputed_data_path=None,
            oracle_arch="t5",
            oracle_size="base",
            save_dir=f"results/llama3_ft/checkpointed/{training_args.estring}",
            oracle_eval_batch_size=16,
            m1_eval_batch_size=64,
            alexpaca_path=output_path,
            # defaults
            m2_eval_batch_size=1,
            downsample_pt_size=None,
            ds_shift=0,
            oai_cache_path=None,
            gt_subset=False,
            results_filename="RESULTS_FILENAME",
        )

if __name__ == "__main__":
    train()

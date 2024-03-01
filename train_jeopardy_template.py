# %%

import transformers
from transformers import BitsAndBytesConfig
from typing import List

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)

import torch
import pandas as pd
from datasets import Dataset

from secondary_model import Alpaca_Secondary_Model

CUTOFF_LEN = 256


alpaca = Alpaca_Secondary_Model(
    "alpaca",
    ".model_cache/alpaca/tuned",
    # precision="bnb_4",
    precision="bf16",
)
alpaca.model = alpaca.model

tokenizer = alpaca.tokenizer

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"

custom_data = Dataset.from_pandas(
    pd.read_hdf("data/jeopardy/jeopardy_full_validation.hd5")
)
custom_data = custom_data.shuffle(seed=42).select(range(400))
custom_data = custom_data.filter(lambda x: x["jeopardy_q"] is not None)


def get_model_size(model):
    sizes = {}
    for layer in model.modules():
        for name, param in layer.named_parameters():
            dtype = param.dtype
            if dtype not in sizes:
                sizes[dtype] = 0
            sizes[dtype] += param.numel() * param.element_size()
    print(sizes)


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


ds = custom_data.map(lambda x: {"prompt": alpaca.fit_template(x["q1"], x["fc_masked"])})
custom_train_val = ds.train_test_split(test_size=200, shuffle=True, seed=42)
train_ds = Dataset.from_list([tokenize(x) for x in custom_train_val["train"]["prompt"]])
val_ds = Dataset.from_list([tokenize(x) for x in custom_train_val["test"]["prompt"]])

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 128
# BATCH_SIZE = 64
# MICRO_BATCH_SIZE = 4
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300
OUTPUT_DIR = "experiments"


alpaca.model = prepare_model_for_int8_training(alpaca.model)
# alpaca.model = prepare_model_for_kbit_training(alpaca.model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
alpaca.model = get_peft_model(alpaca.model, config)
alpaca.model.print_trainable_parameters()


training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard",
)


data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)


trainer = transformers.Trainer(
    model=alpaca.model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=training_arguments,
    data_collator=data_collator,
)
alpaca.model.config.use_cache = False
old_state_dict = alpaca.model.state_dict
alpaca.model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(alpaca.model, type(alpaca.model))

alpaca.model = torch.compile(alpaca.model)

trainer.train()
alpaca.model.save_pretrained(OUTPUT_DIR)

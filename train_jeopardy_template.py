import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from secondary_model import Alpaca_Secondary_Model
import pandas as pd
from datasets import Dataset
from pathlib import PurePath
from datetime import datetime


# The model that you want to train from the Hugging Face hub
# model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
# dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
# new_model = "llama-2-7b-miniguanaco"

# Number of training epochs
# num_train_epochs = 1
num_train_epochs = 200

now = datetime.now()
new_model = f"alpaca-jeopardy"

# Output directory where the model predictions and checkpoints will be stored
output_dir = f"./models/{new_model}"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
# lora_r = 64
lora_r = 8

# Alpha parameter for LoRA scaling
lora_alpha = 32

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

fp16 = False  # Enable fp16/bf16 training (set bf16 to True with an A100)
bf16 = False

per_device_train_batch_size = 12  # Batch size per GPU for training
per_device_eval_batch_size = 3

gradient_accumulation_steps = (
    1  # Number of update steps to accumulate the gradients for
)
gradient_checkpointing = True  # Enable gradient checkpointing
max_grad_norm = 0.3  # Maximum gradient normal (gradient clipping)

# learning_rate = 2e-4  # Initial learning rate (AdamW optimizer)
learning_rate = 2e-4  # testing
weight_decay = (
    0.001  # Weight decay to apply to all layers except bias/LayerNorm weights
)

optim = "paged_adamw_32bit"  # Optimizer to use
lr_scheduler_type = "cosine"  # Learning rate schedule

max_steps = -1  # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03  # Ratio of steps for a linear warmup (from 0 to learning rate)

group_by_length = True  # Group sequences into batches with same length. Saves memory and speeds up training considerably

save_steps = 1000  # Save checkpoint every X updates steps
logging_steps = 25  # Log every X updates steps


################################################################################
# SFT parameters
################################################################################

# # Maximum sequence length to use
# max_seq_length = None
# custom dataset
CUTOFF_LEN = 512

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

alpaca = Alpaca_Secondary_Model(
    "alpaca",
    ".model_cache/alpaca/tuned",
    precision="bnb_4",
    # precision="bf16",
    quantization_config=bnb_config,
)

alpaca.model.eval()

alpaca.model.config.use_cache = False
alpaca.model.config.pretraining_tp = 1

# Load LLaMA tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer = alpaca.tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# train data
# custom_data_train = Dataset.from_pandas(
#     pd.read_json("data/jeopardy/jeopardy_full_train.jsonl", lines=True)
# )
# custom_data_train = custom_data_train.filter(lambda x: x["jeopardy_q"] is not None)
# ds_train = custom_data_train.map(
#     lambda x: {"prompt": alpaca.fit_template(x["q1"], x["fc_masked"])}
# )
# ds_train = ds_train.map(
#     lambda x: {
#         "training_example": x["prompt"]
#         + " "
#         + x["jeopardy_q"]
#         + " "
#         + tokenizer.eos_token
#     }
# )


# # test data
custom_data_test = Dataset.from_pandas(
    pd.read_hdf(
        "data/jeopardy/jeopardy_full_validation.hd5"
    )  # TODO: switch to actual train once processed
)
custom_data_test = custom_data_test.filter(lambda x: x["jeopardy_q"] is not None)
ds_test = custom_data_test.map(
    lambda x: {"prompt": alpaca.fit_template(x["q1"], x["fc_masked"])}
)
ds_test = ds_test.map(
    lambda x: {
        "training_example": x["prompt"]
        + " "
        + x["jeopardy_q"]
        + " "
        + tokenizer.eos_token
    }
)


# downsample (prototyping)
ds_test = ds_test.shuffle(seed=42).select(range(per_device_train_batch_size * 2))
ds_train = ds_test.shuffle(seed=42).select(range(per_device_train_batch_size * 2))

eval_steps = len(ds_train) // per_device_train_batch_size
# eval_steps = 2

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj",
        # "lm_head",
    ],
    # modules_to_save=["embed_tokens"],
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    # num_train_epochs=num_train_epochs
    # / len(ds_train)
    # * 3,  # testing, force 1 train step
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    learning_rate=learning_rate,
    # learning_rate=0, # testing
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
)


def compute_perplexity(eval_preds):
    logits = torch.tensor(eval_preds.predictions)
    batch_size, seq_length, vocab_size = logits.shape
    labels = torch.tensor(eval_preds.label_ids)
    # logits = torch.nn.functional.softmax(logits, dim=-1)
    # p_true_tokens = logits.view(-1, vocab_size)[
    #     torch.arange(batch_size * seq_length), labels.view(-1)
    # ].view(batch_size, seq_length)

    # nll = -torch.log(p_true_tokens + 1e-10)
    # mean_nll = nll.mean()
    # perplexity = torch.exp(mean_nll)

    l2 = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
    ppl = torch.exp(l2)
    # compute percentage of correct tokens
    correct_tokens = (logits.argmax(-1) == labels).float().mean()

    return {"perplexity": ppl, "correct_tokens": correct_tokens.item()}


# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=alpaca.model,
    train_dataset=ds_train,
    eval_dataset=ds_train,
    peft_config=peft_config,
    dataset_text_field="training_example",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    max_seq_length=CUTOFF_LEN,
    compute_metrics=compute_perplexity,
)


# Train model
trainer.model.cuda()
trainer.train()

trainer.model.save_pretrained(PurePath("models") / new_model)

# eval the model on the first 3 examples and print inputs and outputs
trainer.model.eval()
# for i in range(3):
#     tokenized_input = tokenizer(ds_train[i]["prompt"], return_tensors="pt").to(
#             "cuda"
#         )
#     til = tokenized_input["input_ids"].shape[1]
#     generation = trainer.model.generate(
#         **tokenized_input,
#         max_new_tokens=100,
#         eos_token_id=2,
#         output_scores=True,
#         return_dict_in_generate=True
#     )
#     generated_tokens = generation.sequences[:, til:]
#     generated_seq = tokenizer.decode(generated_tokens.squeeze().tolist())
#     print(f"Input: {ds_train[i]['prompt']}")  # noqa
#     print(f"Output: {generated_sez}")  # noqa

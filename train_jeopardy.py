# %%
# https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=OJXpOgBFuSrc

import main

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    GenerationConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datetime import datetime
import os
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd

from secondary_model import Alpaca_Secondary_Model
import wandb
import os
import argparse

###### Testing Setup ######

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--model_load_path", type=str, default=".model_cache/alpaca/tuned")
args = parser.parse_args()
DEBUG_MODE = args.debug
MODEL_LOAD_PATH = args.model_load_path

if DEBUG_MODE:
    print("DEBUG MODE ENABLED")


######## W&B Setup ########
os.environ["WANDB_PROJECT"] = "qq"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")  # save W&B logs locally
os.environ["WANDB_CACHE_DIR"] = os.path.join(
    os.getcwd(), "wandb/cache"
)  # save W&B cache locally
os.environ["WANDB_CONFIG_DIR"] = os.path.join(
    os.getcwd(), "wandb/config"
)  # save W&B config locally


######## Helper Functions ########


def custom_metrics(eval_preds):
    logits = torch.tensor(eval_preds.predictions)
    labels = torch.tensor(eval_preds.label_ids)
    batch_size, seq_length, vocab_size = logits.shape

    # steal from inside llama
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # loss_fct = CrossEntropyLoss()
    # shift_logits = shift_logits.view(-1, vocab_size)
    # shift_labels = shift_labels.view(-1)

    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
    p_true_tokens = probs.view(-1, vocab_size)[
        torch.arange(batch_size * (seq_length - 1)), shift_labels.view(-1)
    ].view(batch_size, (seq_length - 1))

    nll = -torch.log(p_true_tokens)
    # set likelihoods to 0 for padding tokens
    nll[shift_labels == -100] = 0
    num_non_pad = (shift_labels != -100).sum(axis=1)
    mean_nll = nll.sum(axis=1) / num_non_pad
    ppl = torch.exp(mean_nll).mean()

    # print the first 5 examples
    for i in range(5):
        print(
            f"Example {i}: {tokenizer.decode(shift_labels[i])} | {tokenizer.decode(shift_logits[i].argmax(-1))}"
        )

    # correct_tokens = (shift_logits.argmax(-1) == shift_labels).float().mean()

    return {
        "perplexity": ppl,
    }


######## Main ########

### Model and Tokenizer Setup ###
load_dotenv()
now = datetime.now().strftime("%Y-%m-%d-%H-%M")

alpaca = Alpaca_Secondary_Model(
    "alpaca",
    MODEL_LOAD_PATH,
    # precision="bnb_4",
    # quantization_config=bnb_config,
)
model = alpaca.model
tokenizer = alpaca.tokenizer
tokenizer.padding_side = "left"

# load from checkpoint instead
# model = AutoModelForCausalLM.from_pretrained(model, "./models/checkpoint-100")

# dataset = load_from_disk("datasets/dataset") # custom dataset to confirm model is learning
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, quantization_config=bnb_config, device_map={"": 0}, token=access_token
# )

### Dataset Setup ###

dataset_train = Dataset.from_pandas(
    pd.read_json("data/jeopardy/jeopardy_gpt_full_train.jsonl", lines=True)
)
dataset_train = dataset_train.select(range(20000))
dataset_train = dataset_train.filter(lambda x: x["jeopardy_q"] is not None)
dataset_train = dataset_train.map(
    lambda x: {"prompt": alpaca.fit_template(x["q1"], x["fc_masked"])}
)
dataset_train = dataset_train.map(
    lambda x: {"text": x["prompt"] + " " + x["jeopardy_q"] + " " + tokenizer.eos_token}
)

dataset_val = Dataset.from_pandas(
    pd.read_json("data/jeopardy/jeopardy_full_validation.jsonl", lines=True)
).select(range(1000 if not DEBUG_MODE else 128))
# dataset_val = dataset_val.select(range(4))  # testing
dataset_val = dataset_val.filter(lambda x: x["jeopardy_q"] is not None)
dataset_val = dataset_val.map(
    lambda x: {"prompt": alpaca.fit_template(x["q1"], x["fc_masked"])}
)
dataset_val = dataset_val.map(
    lambda x: {"text": x["prompt"] + " " + x["jeopardy_q"] + " " + tokenizer.eos_token}
)


# %%

### Model Training ###

# Resize the embeddings
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = (
    False  # Gradient checkpointing is used by default but not compatible with caching
)

# model = prepare_model_for_kbit_training(model)
# lora_config = LoraConfig(
#     r=256,
#     lora_alpha=128,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "q_proj",
#         "v_proj",
#         "k_proj",
#         "o_proj",
#         "gate_proj",
#         "down_proj",
#         "up_proj",
#     ],
# )

# model = get_peft_model(model, lora_config)


def generate(prompt):
    # prompt = "### Human: " + instruction + "### Assistant: "
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=0.0, top_p=1.0, top_k=50, num_beams=1
        ),
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=50,
        # pad_token_id=tokenizer.pad_token_id # For some reason, needed to allow inference to work if using saved 8bit model
    )
    for seq in generation_output.sequences:
        output = tokenizer.decode(seq)
        print(output)


# %%
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)


class CustomCollator(DataCollatorForCompletionOnlyLM):
    def __call__(self, examples):
        batch = super().__call__(examples)
        labels = batch["labels"].clone()
        # reset the last token to 2
        labels[:, -1] = 2
        batch["labels"] = labels
        return batch


# data_collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
response_template = "Response:"
data_collator = collator = CustomCollator(response_template, tokenizer=tokenizer)
# %%

run_name = "alpaca-jeopardy"
training_arguments = TrainingArguments(
    output_dir=f"./models/alexpaca/{str(now)}",
    evaluation_strategy="steps",
    do_eval=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    log_level="debug",
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    save_steps=1,
    # save_steps=3,
    logging_steps=500,
    logging_first_step=True,
    save_total_limit=2,
    learning_rate=(2e-5),
    eval_steps=1000 if not DEBUG_MODE else 64,
    # eval_steps=1,  # testing
    max_grad_norm=0.3,
    num_train_epochs=1,
    # max_steps=2,  # testing
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="wandb",
    run_name=f"{run_name}-{now}",
    eval_accumulation_steps=1,
)
wandb.init(project="qq", name=f"{run_name}-{now}", config=training_arguments)
# os.environ["WANDB_DISABLED"] = "true"
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    # peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    compute_metrics=custom_metrics,
    data_collator=data_collator,
)

# %%
eval_results = trainer.evaluate()
trainer.train()
for i in range(3):
    generate(dataset_val["text"][i].split("### Response: ")[0] + "### Response: ")

save_path = f"./models/alexpaca/{now}"
model.save_pretrained(save_path)

main.main(
    split="validation",
    m1_arch="t5-base",
    oracle_arch="t5",
    oracle_size="base",
    m2_arch="alexpaca",
    alexpaca_path=save_path,
    save_dir="results/3_alexpaca/main",
    # defaults
    m1_path=None,
    template_id=None,
    m1_eval_batch_size=1,
    oracle_eval_batch_size=1,
    downsample_pt_size=None,
    ds_shift=0,
    oai_cache_path=None,
    results_filename=None,
    gt_subset=False,
)

# %%

# %%
# https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=OJXpOgBFuSrc


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
)
from trl import SFTTrainer
from datetime import datetime
import os
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd

from secondary_model import Alpaca_Secondary_Model

# If you prefer to create/use an 8bit version of the model for faster loading instead, create/save it using the following code
# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map={'': 0}, load_in_8bit=True)
# model.save_pretrained('meta-llama-Llama-2-7b-hf-CausalLM-8bit') # save_pretrained is not currently supported for 4bit model
# model_name = "meta-llama-Llama-2-7b-hf-CausalLM-8bit" # Saved 8bit model loads in 2 min, ~5x faster. Takes 50% more memory and 50% longer to train

######## Helper Functions ########


def compute_perplexity(eval_preds):
    logits = torch.tensor(eval_preds.predictions)
    labels = torch.tensor(eval_preds.label_ids)
    batch_size, seq_length, vocab_size = logits.shape

    # steal from inside llama
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
    p_true_tokens = probs.view(-1, vocab_size)[
        torch.arange(batch_size * (seq_length - 1)), shift_labels.view(-1)
    ].view(batch_size, (seq_length - 1))

    nll = -torch.log(p_true_tokens)
    mean_nll = nll.mean()
    ppl = torch.exp(mean_nll)

    correct_tokens = (shift_logits.argmax(-1) == shift_labels).float().mean()

    return {"perplexity": ppl, "correct_tokens": correct_tokens.item()}


######## Main ########

### Model and Tokenizer Setup ###
load_dotenv()
now = datetime.now().strftime("%Y-%m-%d-%H-%M")
# model_name = "meta-llama/Llama-2-7b-hf"
# access_token = os.getenv("ACCESS_TOKEN")
# tokenizer_model_name = (
#     "meta-llama/Llama-2-7b-hf"  # Tokenizer (not saved with 8bit model)
# )
# tokenizer = AutoTokenizer.from_pretrained(
# tokenizer_model_name, use_fast=True, token=access_token
# )
# Create a new token and add it to the tokenizer
# tokenizer.add_special_tokens({"pad_token": "<pad>"})
# compute_dtype = getattr(torch, "float16")
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=True,
# )
alpaca = Alpaca_Secondary_Model(
    "alpaca",
    ".model_cache/alpaca/tuned",
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
    pd.read_json("data/jeopardy/jeopardy_full_train.jsonl", lines=True)
)
dataset_train = dataset_train.filter(lambda x: x["jeopardy_q"] is not None)
dataset_train = dataset_train.map(
    lambda x: {"prompt": alpaca.fit_template(x["q1"], x["fc_masked"])}
)
dataset_train = dataset_train.map(
    lambda x: {"text": x["prompt"] + " " + x["jeopardy_q"] + " " + tokenizer.eos_token}
)

dataset_val = Dataset.from_pandas(
    pd.read_json("data/jeopardy/jeopardy_full_validation.jsonl", lines=True)
)
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
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)


def generate(prompt):
    # prompt = "### Human: " + instruction + "### Assistant: "
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=1.0, top_p=1.0, top_k=50, num_beams=1
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

# %%

run_name = "alpaca-jeopardy"
training_arguments = TrainingArguments(
    output_dir="./models",
    evaluation_strategy="steps",
    # evaluation_strategy="no",
    do_eval=True,
    per_device_train_batch_size=8,  # 8 works, but not faster on 4070
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
    log_level="debug",
    optim="paged_adamw_32bit",
    save_steps=10000,  # change to 500
    logging_steps=1000,  # change to 100
    logging_first_step=True,
    save_total_limit=2,
    learning_rate=1e-3,
    eval_steps=100,  # change to 200
    # bf16=True, # Ampere+ architecture, comment out on non-Ampere+
    max_grad_norm=0.3,
    num_train_epochs=1,
    # max_steps=100,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    run_name=f"{run_name}-{now}",  # Name of the W&B run (optional)
)

os.environ["WANDB_DISABLED"] = "true"
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    # peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    compute_metrics=compute_perplexity,
)

for i in range(3):
    generate(dataset_val["text"][i].split("### Response: ")[0] + "### Response: ")
trainer.train()
for i in range(3):
    generate(dataset_val["text"][i].split("### Response: ")[0] + "### Response: ")

# %%

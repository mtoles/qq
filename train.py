import datetime
import numpy as np
import click
import torch
from datasets import load_dataset

from utils import get_downsample_dataset_size_str

from transformers import (
    BigBirdTokenizer,
    Trainer,
    TrainingArguments,
)


from bb_model import BigBirdForNaturalQuestions, collate_fn, compute_metrics


# TRAIN_ON_SMALL = os.environ.pop("TRAIN_ON_SMALL", "false")
RESUME_TRAINING = None

# os.environ["WANDB_WATCH"] = "false"
# os.environ["WANDB_PROJECT"] = "bigbird-natural-questions"
SEED = 42
GROUP_BY_LENGTH = True
LEARNING_RATE = 3.0e-5
WARMUP_STEPS = 100
MAX_EPOCHS = 9
FP16 = False
SCHEDULER = "linear"
MODEL_ID = "google/bigbird-roberta-base"


@click.command()

# model
@click.option("--model_path", default=None, help="path to mdoel")
@click.option("--mode", default="train", help="{train | eval}")

# data
@click.option("--dataset", help="{natural_questions | hotpot}")
@click.option("--batch_size", default=1, type=int, help="batch size")
@click.option(
    "--downsample_data_size_val",
    default=None,
    help="use at most this many examples in validation",
)
@click.option(
    "--downsample_data_size_train",
    default=None,
    help="use at most this many examples in training",
)

# distributed training
@click.option(
    "--local_rank",
    default=0,
    type=int,
    help="local_rank for distributed training on gpus",
)
@click.option("--nproc_per_node", default=1, type=int, help="num of processes per node")
@click.option("--master_port", default=1234, type=int, help="master port")
def main(
    model_path,
    mode,
    dataset,
    batch_size,
    downsample_data_size_val,
    downsample_data_size_train,
    local_rank,
    nproc_per_node,
    master_port,
):
    # "nq-training.jsonl" & "nq-validation.jsonl" are obtained from running `prepare_nq.py`
    # prepare training run for multiple GPUs:
    # if local_rank != -1:
    #     torch.cuda.set_device(local_rank)
    #     device = torch.device("cuda", local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method="env://")
    assert mode in ["train", "eval"], f"mode {mode} not supported"
    assert dataset in [
        "natural_questions",
        "hotpot",
    ], f"dataset {dataset} not supported"

    if dataset == "natural_questions":
        tr_dataset = load_dataset("json", data_files="data/nq-training.jsonl", split=f"train{get_downsample_dataset_size_str(downsample_data_size_train)}")
        val_dataset = load_dataset("json", data_files="data/nq-validation.jsonl", split=f"train{get_downsample_dataset_size_str(downsample_data_size_val)}")
        output_dir = "bigbird-nq-complete-tuning"
    elif dataset == "hotpot":
        tr_dataset = load_dataset(
            "json",
            data_files="data/hotpot-training.jsonl", split=f"train{get_downsample_dataset_size_str(downsample_data_size_train)}"
        )
        val_dataset = load_dataset("json", data_files="data/hotpot-validation.jsonl", split=f"train{get_downsample_dataset_size_str(downsample_data_size_val)}")
        output_dir = "bigbird-hotpot-complete-tuning"

    # tr_dataset = downsample(tr_dataset, downsample_data_size_train)
    # val_dataset = downsample(val_dataset, downsample_data_size_val)

    print(tr_dataset, val_dataset)

    tokenizer = BigBirdTokenizer.from_pretrained(MODEL_ID)
    if model_path is not None:
        model = BigBirdForNaturalQuestions.from_pretrained(model_path)
    else:
        model = BigBirdForNaturalQuestions.from_pretrained(
            MODEL_ID, gradient_checkpointing=True
        )

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        per_gpu_train_batch_size=batch_size,
        per_gpu_eval_batch_size=batch_size,
        per_device_train_batch_size=2 * batch_size,
        gradient_accumulation_steps=10,
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=SCHEDULER,
        num_train_epochs=MAX_EPOCHS,
        run_name=f"bigbird-{dataset}-complete-tuning-exp",
        disable_tqdm=False,
        # load_best_model_at_end=True,
        # report_to="wandb",
        remove_unused_columns=False,
        fp16=FP16,
        label_names=[
            "pooler_label",
            "start_positions",
            "end_positions",
        ],  # it's important to log eval_loss
        evaluation_strategy="epoch",
        eval_steps=0.05,
        save_strategy="epoch",
        save_steps=0.05,
        logging_strategy="epoch",
        logging_steps=0.05,
        logging_dir="tb_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        report_to="tensorboard",
        logging_first_step=True,
    )
    print("Batch Size", args.train_batch_size)
    print("Parallel Mode", args.parallel_mode)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=tr_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda x: compute_metrics(tokenizer, x),
    )
    if mode == "eval":
        metrics = trainer.evaluate()
        print(metrics)
        return
    elif mode == "train":
        try:
            trainer.train(resume_from_checkpoint=RESUME_TRAINING)
            trainer.save_model(f"{dataset}-final-model-exp")
        except KeyboardInterrupt:
            trainer.save_model(f"{dataset}-interrupted-model-exp")
        # wandb.finish()


if __name__ == "__main__":
    main()

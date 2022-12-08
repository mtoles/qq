import datetime
import numpy as np
import click
from datasets import load_dataset

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
MAX_EPOCHS = 5
FP16 = False
SCHEDULER = "linear"
MODEL_ID = "google/bigbird-roberta-base"


@click.command()
@click.option("--train_on_small", default="false", help="Use small dataset")
@click.option("--dataset", help="{natural_questions | hotpot}")
@click.option(
    "--local_rank",
    default=-1,
    type=int,
    help="local_rank for distributed training on gpus",
)
@click.option("--nproc_per_node", default=1, type=int, help="num of processes per node")
@click.option("--batch_size", default=1, type=int, help="batch size")
def main(train_on_small, dataset, batch_size, local_rank, nproc_per_node):
    # "nq-training.jsonl" & "nq-validation.jsonl" are obtained from running `prepare_nq.py`

    if dataset == "natural_questions":
        tr_dataset = load_dataset("json", data_files="data/nq-training.jsonl")["train"]
        val_dataset = load_dataset("json", data_files="data/nq-validation.jsonl")[
            "train"
        ]
        output_dir = "bigbird-nq-complete-tuning"
    elif dataset == "hotpot":
        tr_dataset = load_dataset("json", data_files="data/hotpot-training.jsonl")[
            "train"
        ]
        val_dataset = load_dataset("json", data_files="data/hotpot-validation.jsonl")[
            "train"
        ]
        output_dir = "bigbird-hotpot-complete-tuning"

    else:
        raise ValueError(f"dataset {dataset} not supported")

    if train_on_small == "true":
        # this will run for ~12 hrs on 2 K80 GPU (natural questions)
        np.random.seed(SEED)
        indices = np.random.randint(0, 298152, size=8000)
        tr_dataset = tr_dataset.select(indices)
        np.random.seed(SEED)
        indices = np.random.randint(0, 9000, size=1000)
        val_dataset = val_dataset.select(indices)

    print(tr_dataset, val_dataset)

    tokenizer = BigBirdTokenizer.from_pretrained(MODEL_ID)
    model = BigBirdForNaturalQuestions.from_pretrained(
        MODEL_ID, gradient_checkpointing=True
    )

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
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
        evaluation_strategy="steps",
        eval_epochs=5,
        save_strategy="epochs",
        save_steps=1,
        logging_strategy="steps",
        logging_steps=5,
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
        compute_metrics=compute_metrics,
    )
    try:
        trainer.train(resume_from_checkpoint=RESUME_TRAINING)
        trainer.save_model(f"{dataset}-final-model-exp")
    except KeyboardInterrupt:
        trainer.save_model("interrupted-natural-questions")
    # wandb.finish()


if __name__ == "__main__":
    main()

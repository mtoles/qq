import datetime
import click
from datasets import load_from_disk

from prepare_data import prepare_inputs_hp, prepend_question
from utils import BB_MODEL_ID, collate_fn_bb, check_tokenizer
from dataset_utils import drop_unanswerable, check_dataset
from metrics import compute_metrics_bb
from bb_model import BigBirdForNaturalQuestions

from transformers import (
    BigBirdTokenizer,
    Trainer,
    TrainingArguments,
)


RESUME_TRAINING = None
SEED = 42
GROUP_BY_LENGTH = True
FP16 = False
SCHEDULER = "linear"


@click.command()

# model
@click.option("--model_path", default=None, help="path to model")
@click.option("--model_class", help="model name { bigbird }")
@click.option("--mode", default="train", help="{train | eval}")
@click.option("--learning_rate", type=float, help="learning rate")
@click.option("--max_epochs", type=int, help="max epochs")
@click.option(
    "--gradient_accumulation_steps", type=int, help="gradient accumulation steps"
)
@click.option("--warmup_steps", type=int, help="warmup steps")

# data
@click.option(
    "--tr_dataset_path", help="path to train {natural_questions | hotpot} dataset"
)
@click.option(
    "--val_dataset_path", help="path to validation {natural_questions | hotpot} dataset"
)
@click.option("--tr_batch_size", default=1, type=int, help="batch size")
@click.option("--eval_batch_size", default=16, type=int, help="eval batch size")
@click.option(
    "--downsample_data_size_val",
    default=None,
    type=int,
    help="use at most this many examples in validation",
)
@click.option(
    "--downsample_data_size_train",
    default=None,
    type=int,
    help="use at most this many examples in training",
)
@click.option(
    "--masking_scheme",
    default="None",
    help="which context column to use. {'None' | 'randomsentence'}",
)
@click.option(
    "--load_from_cache",
    type=bool,
    default=False,
    help="use huggingface cache for data load, map, and filter",
)
@click.option(
    "--log_eval",
    default=False,
    help="write eval metrics and examples to log file in inf_logs/",
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
    model_class,
    mode,
    tr_batch_size,
    eval_batch_size,
    learning_rate,
    max_epochs,
    gradient_accumulation_steps,
    warmup_steps,
    tr_dataset_path,
    val_dataset_path,
    downsample_data_size_val,
    downsample_data_size_train,
    masking_scheme,
    load_from_cache,
    log_eval,
    local_rank,
    nproc_per_node,
    master_port,
):
    # Unit Tests

    assert mode in ["train", "eval"], f"mode {mode} not supported"
    assert val_dataset_path is not None, "val_dataset_path must be specified"
    if "natural_questions" in val_dataset_path:
        base_dataset = "natural_questions"
    elif "hotpot" in val_dataset_path:
        base_dataset = "hotpot"
    else:
        raise ValueError(
            f"dataset {val_dataset_path} does not contain 'natural_questions' or 'hotpot'"
        )

    # Load Model

    if model_class == "bigbird":
        MODEL_ID = BB_MODEL_ID
        tk = BigBirdTokenizer.from_pretrained(MODEL_ID)
        if model_path is not None:
            model = BigBirdForNaturalQuestions.from_pretrained(model_path, tk)
        else:
            model = BigBirdForNaturalQuestions.from_pretrained(
                MODEL_ID, gradient_checkpointing=True, tk=tk
            )
    else:
        raise ValueError(f"model {model_class} not supported")
    output_dir = f"bigbird_{base_dataset}_complete_tuning"
    check_tokenizer(tk)
    max_length = model.bert.embeddings.position_embeddings.weight.shape[0]
    # Load Data
    if tr_dataset_path is not None:
        tr_dataset = load_from_disk(tr_dataset_path)
        if downsample_data_size_train is not None:
            tr_dataset = tr_dataset.select(range(downsample_data_size_train))
        # Drop examples that do not have answer in the context
        tr_dataset = drop_unanswerable(tr_dataset, masking_scheme, load_from_cache)

        print("Prepending questions")
        tr_dataset = tr_dataset.map(
            lambda x: prepend_question(
                x, masking_scheme=masking_scheme, sep_token=tk.sep_token
            ),
            load_from_cache_file=load_from_cache,
        )

        print("Preparing train inputs hotpot...")
        tr_dataset = tr_dataset.map(
            lambda x: prepare_inputs_hp(
                x,
                tk=tk,
                max_length=max_length,
                masking_scheme=masking_scheme,
            ),
            load_from_cache_file=load_from_cache,
        )
        # check_dataset(tr_dataset, tokenizer)
    else:
        tr_dataset = None

    val_dataset = load_from_disk(val_dataset_path)
    if downsample_data_size_val is not None:
        val_dataset = val_dataset.select(range(downsample_data_size_val))
    print("Dropping val unanswerable...")
    val_dataset = drop_unanswerable(val_dataset, masking_scheme, load_from_cache)
    print("Prepending val questions...")
    val_dataset = val_dataset.map(
        lambda x: prepend_question(
            x, masking_scheme=masking_scheme, sep_token=tk.sep_token
        ),
        load_from_cache_file=load_from_cache,
    )
    print("Preparing val inputs...")
    val_dataset = val_dataset.map(
        lambda x: prepare_inputs_hp(
            x, tk=tk, max_length=max_length, masking_scheme=masking_scheme
        ),
        load_from_cache_file=load_from_cache,
    )
    # check_dataset(val_dataset, tokenizer)

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        # per_gpu_train_batch_size=batch_size,
        per_gpu_eval_batch_size=eval_batch_size,
        per_device_train_batch_size=tr_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=SCHEDULER,
        num_train_epochs=max_epochs,
        run_name=f"bigbird-{base_dataset}-complete-tuning-exp",
        disable_tqdm=False,
        # load_best_model_at_end=True,
        remove_unused_columns=False,
        fp16=FP16,
        label_names=[
            "pooler_label",
            "start_positions",
            "end_positions",
            "gt_answers",
        ],
        evaluation_strategy="epoch",
        eval_steps=0.05,
        save_strategy="epoch",
        save_steps=0.05,
        # logging_strategy="epoch",
        logging_steps=10,
        logging_dir="tb_logs/" + now,
        report_to="tensorboard",
        logging_first_step=True,
    )
    print("Batch Size", args.train_batch_size)
    print("Parallel Mode", args.parallel_mode)

    log_path = "inf_logs/" + now if log_eval else None
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=(lambda x: collate_fn_bb(x, tk)),
        train_dataset=tr_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda x: compute_metrics_bb(x, tk, log_path),
        tokenizer=tk,
    )
    if mode == "eval":
        metrics = trainer.evaluate()
        print(metrics)
    elif mode == "train":
        try:
            # trainer.train(resume_from_checkpoint=RESUME_TRAINING)
            trainer.evaluate()
            trainer.train(model_path)
            trainer.save_model(f"models/{base_dataset}-final-model-exp-{now}")
        except KeyboardInterrupt:
            trainer.save_model(f"models/{base_dataset}-interrupted-model-exp-{now}")


if __name__ == "__main__":
    main()

import datetime
import numpy as np
import click
import torch
from datasets import load_from_disk

from prepare_data import prepare_inputs_hp
from utils import BB_MODEL_ID, collate_fn, check_tokenizer
from dataset_utils import drop_unanswerable, check_dataset
from metrics import compute_metrics
from bb_model import BigBirdForNaturalQuestions

from transformers import (
    BigBirdTokenizer,
    Trainer,
    TrainingArguments,
)


# TRAIN_ON_SMALL = os.environ.pop("TRAIN_ON_SMALL", "false")
RESUME_TRAINING = None

# os.environ["WANDB_WATCH"] = "false"
# os.environ["WANDB_PROJECT"] = "bigbird-natural-questions"
SEED = 42
GROUP_BY_LENGTH = True
# LEARNING_RATE = 3.0e-5
# MAX_EPOCHS = 9
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
@click.option("--eval_batch_size", default=64, type=int, help="eval batch size")
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
        tokenizer = BigBirdTokenizer.from_pretrained(MODEL_ID)
        if model_path is not None:
            # todo: need tokenizer from somewhere
            model = BigBirdForNaturalQuestions.from_pretrained(model_path, tokenizer)
        else:
            model = BigBirdForNaturalQuestions.from_pretrained(
                MODEL_ID, gradient_checkpointing=True, tk=tokenizer
            )
    else:
        raise ValueError(f"model {model_class} not supported")
    output_dir = f"bigbird_{base_dataset}_complete_tuning"
    check_tokenizer(tokenizer)
    max_length = model.bert.embeddings.position_embeddings.weight.shape[0]
    # Load Data
    if tr_dataset_path is not None:
        tr_dataset = load_from_disk(tr_dataset_path)
        if downsample_data_size_train is not None:
            tr_dataset = tr_dataset.select(range(downsample_data_size_train))
        # Drop examples that do not have answer in the context
        # Should drop 6 examples from train
        # tr_dataset = tr_dataset.filter(lambda x: x["id"]=="5a828cd455429940e5e1a8f1")
        tr_dataset = drop_unanswerable(tr_dataset, masking_scheme, load_from_cache)

        # tr_dataset = tr_dataset.select(range(54000, 55000))  # testing
        print("Preparing train inputs hotpot...")
        tr_dataset = tr_dataset.map(
            lambda x: prepare_inputs_hp(
                x,
                tokenizer=tokenizer,
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
    print("Preparing validation inputs hotpot...")
    val_dataset = drop_unanswerable(val_dataset, masking_scheme, load_from_cache)
    val_dataset = val_dataset.map(
        lambda x: prepare_inputs_hp(
            x, tokenizer=tokenizer, max_length=max_length, masking_scheme=masking_scheme
        ),
        load_from_cache_file=load_from_cache,
    )
    # check_dataset(val_dataset, tokenizer)

    # # test some examples
    # for i in range(5):
    #     ex = val_dataset[i]
    #     st = val_dataset[i]["labels"]["start_token"]
    #     et = val_dataset[i]["labels"]["end_token"]
    #     lab = ex["input_ids"][st:et]
    #     print(f"processed: {lab} | raw: {ex['answer']}")

    # print

    # tr_dataset = (
    #     load_dataset(
    #         "json",
    #         data_files=tr_dataset_path,
    #         split=f"train{get_downsample_dataset_size_str(downsample_data_size_train)}",
    #         download_mode="force_redownload",
    #     )
    #     if mode == "train"
    #     else None
    # )
    # val_dataset = load_dataset(
    #     "json",
    #     data_files=val_dataset_path,
    #     split=f"train{get_downsample_dataset_size_str(downsample_data_size_val)}",
    #     download_mode="force_redownload",
    # )

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
        ],  # it's important to log eval_loss
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
        data_collator=(lambda x: collate_fn(x, tokenizer)),
        train_dataset=tr_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, log_path),
        tokenizer=tokenizer,  # experimental... trying to get pad_token_id==0, not ==-100
    )
    if mode == "eval":
        metrics = trainer.evaluate()
        print(metrics)
    elif mode == "train":
        try:
            # trainer.train(resume_from_checkpoint=RESUME_TRAINING)
            trainer.train(model_path)
            trainer.save_model(f"models/{base_dataset}-final-model-exp-{now}")
        except KeyboardInterrupt:
            trainer.save_model(f"models/{base_dataset}-interrupted-model-exp-{now}")
        # wandb.finish()


if __name__ == "__main__":
    main()

# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
from datasets import load_from_disk
from oracles import T5_Bool_Oracle
from secondary_model import (
    Repeater_Secondary_Model,
    OpenAI_Secondary_Model,
    Gt_Secondary_Model,
)
from primary_models import BigBird_PM, T5_PM
from dataset_utils import drop_unanswerable
from datetime import datetime

import pandas as pd


@click.command()
@click.option("--pt_dataset_path", help="path to primary task dataset")
@click.option("--m1_path", help="path to primary model")
@click.option("--m1_arch", help="primary model architecture")
@click.option("--m2_arch", help="secondary model architecture")
@click.option("--oracle_arch", help="oracle architecture")
@click.option("--eval_batch_size", default=2, help="batch size for eval")
@click.option("--masking_scheme", help="{randomsentence | None")
@click.option(
    "--downsample_pt_size",
    default=None,
    help="use at most this many examples in validation",
)
@click.option("--results_filename", help="path to save results")
@click.option(
    "--profile_only",
    is_flag=True,
    default=False,
    help="only profile the primary model on dataset, then exit",
)
def main(
    pt_dataset_path,
    m1_path,
    m1_arch,
    m2_arch,
    oracle_arch,
    eval_batch_size,
    masking_scheme,
    downsample_pt_size,
    results_filename,
    profile_only,
):
    # df = pd.read_csv("None_profile.csv")
    # df = df[
    #     [
    #         "q1",
    #         "a1",
    #         "masked_sentence",
    #         "fc_randomsentence",
    #         "m1_randomsentence_None_gen",
    #         "m1_randomsentence_None_f1",
    #         "m1_supporting_None_gen",
    #         "m1_supporting_None_f1",
    #     ]
    # ]
    # df.to_csv("None_profile_trimmed.csv", index=False)
    now = datetime.now().strftime("Y%m%d-%H%M%S")
    if results_filename is None:
        results_filename = f"{m1_arch}-{downsample_pt_size}-{masking_scheme}-{now}"

    with open(f"inf_logs/{results_filename}.txt", "a") as f:

        masking_str = f"fc_{masking_scheme}"

        # Unit Tests
        assert m1_arch in [
            "bigbird",
            "t5-small",
            "t5-base",
            "t5-large",
            "t5-xl",
            "t5-xxl",
        ]
        assert m2_arch in ["repeater", "openai", "gt"]
        assert oracle_arch.startswith("t5") or oracle_arch == "dummy"

        # Receive and prepare the primary task
        raw_dataset = load_from_disk(pt_dataset_path)
        original_raw_dataset_len = len(raw_dataset)
        if str(downsample_pt_size) != "None":
            raw_dataset = raw_dataset.select(range(int(downsample_pt_size)))

        # Load primary model
        if m1_arch == "bigbird":
            m1 = BigBird_PM(m1_path, eval_batch_size=eval_batch_size)
        elif m1_arch.startswith("t5"):
            m1 = T5_PM(
                eval_batch_size=eval_batch_size,
                model_name=m1_arch,
            )
        else:
            raise NotImplementedError

        ds = raw_dataset

        # Evaluate the primary model
        metrics = {}
        ds, metrics[masking_scheme] = m1.evaluate(
            masking_scheme=masking_scheme, ds=ds, a2_col=None
        )
        ds, metrics["supporting"] = m1.evaluate(
            masking_scheme="supporting", ds=ds, a2_col=None
        )
        if profile_only:
            df = pd.DataFrame(ds)
            df.to_csv(f"{downsample_pt_size}_profile.csv")
            quit()
        # Create the secondary model
        if m2_arch == "repeater":
            m2 = Repeater_Secondary_Model()
        elif m2_arch == "openai":
            m2 = OpenAI_Secondary_Model()
        elif m2_arch == "gt":
            m2 = Gt_Secondary_Model()
        else:
            raise NotImplementedError(f"m2_arch {m2_arch} not implemented")
        # Apply the secondary model
        ds = m2.process(
            ds,
            q1_col="q1",
            masking_scheme=masking_scheme,
        )
        # Save memory by moving m1 to CPU
        m1.model.cpu()
        # Create the oracle
        oracle = T5_Bool_Oracle(model_name=oracle_arch)
        # Answer questions with the oracle
        ds = oracle.process(ds, q2_masking_scheme=masking_scheme)
        oracle.model.cpu()
        m1.model.cuda()
        ds, metrics["answered"] = m1.evaluate(
            masking_scheme=masking_scheme, ds=ds, a2_col="a2"
        )

        # Analysis
        df = pd.DataFrame(ds)
        percent_oracle_correct = df[f"a2_is_correct_{masking_scheme}"].mean()
        print(metrics)

        f.write(
            f"""Model: {m1_path if m1_path else m1_arch}
Masking Scheme:  {masking_scheme}
Oracle:          {oracle.model_name}
Datetime:        {now}
Data:            {pt_dataset_path} {original_raw_dataset_len}/{len(raw_dataset)}
Masking:         {masking_scheme}
F1 delta:        {metrics["answered"]["f1"]-metrics[masking_scheme]["f1"]}
Precision delta: {metrics["answered"]["precision"]-metrics[masking_scheme]["precision"]}
Recall delta:    {metrics["answered"]["recall"]-metrics[masking_scheme]["recall"]}
Oracle acc:      {percent_oracle_correct}
\n
"""
        )


if __name__ == "__main__":
    main()

# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
from datasets import load_from_disk
from oracles import T5_Bool_Oracle
from primary_models import get_m1
from secondary_model import (
    Repeater_Secondary_Model,
    OpenAI_Secondary_Model,
    Gt_Secondary_Model,
)
from dataset_utils import bf_filtering, combine_adversarial_ds
from datetime import datetime
from masking import bf_del_sentences, bf_add_sentences, reduce_to_n
import pandas as pd


@click.command()
@click.option("--pt_dataset_path", help="path to primary task dataset")
@click.option("--m1_path", help="path to primary model")
@click.option("--m1_arch", help="primary model architecture")
@click.option("--m2_arch", help="secondary model architecture")
@click.option("--oracle_arch", help="oracle architecture")
@click.option("--eval_batch_size", default=2, help="batch size for eval")
@click.option("--masking_scheme", help="{randomsentence | bfdelsentence | None")
@click.option("--adversarial_drop_thresh", default=0.5, help="include only examples in the adversarially generated examples where the delta between baseline and masked or distracted is greater than this threshold")
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
    adversarial_drop_thresh,
    downsample_pt_size,
    results_filename,
    profile_only,
):
    ds_masking_scheme = (
        "None" if masking_scheme == "bfdelsentence" else "masking_scheme"
    )
    now = datetime.now().strftime("Y%m%d-%H%M%S")
    if results_filename is None:
        results_filename = f"{m1_arch}-{downsample_pt_size}-{ds_masking_scheme}-{now}"

    with open(f"inf_logs/{results_filename}.txt", "a") as f:
        assert m2_arch in ["repeater", "openai", "gt"]
        assert oracle_arch.startswith("t5") or oracle_arch == "dummy"

        masking_str = f"fc_{masking_scheme}"

        m1 = get_m1(m1_path, m1_arch, eval_batch_size)
        # Receive and prepare the primary task

        raw_dataset = load_from_disk(pt_dataset_path)
        if str(downsample_pt_size) != "None":
            raw_dataset = raw_dataset.select(range(int(downsample_pt_size)))
        original_raw_dataset_len = len(raw_dataset)
        ds = raw_dataset
        # Evaluate the primary model
        metrics = {}

        # first pass
        ds, metrics["supporting"] = m1.evaluate(
            masking_scheme="supporting", ds=ds, a2_col=None
        )

        # select and mask examples where the primary
        if masking_scheme == "bfsentence":
            # if you use {} instead of {"sentences": []} you encounter a bug in datasets.Dataset.map() version 2.10.1
            ds = ds.add_column(
                "context_bfdelsentence", [{"sentences": []} for _ in range(len(ds))]
            )
            ds = ds.add_column(
                "context_bfaddsentence", [{"sentences": []} for _ in range(len(ds))]
            )

            # masking
            ds_got_right_None = ds.filter(lambda x: x["m1_supporting_None_f1"] > 0.0)
            ds_bfdelsentence = bf_del_sentences(ds_got_right_None)  # filtering is wrong

            ds_bfdelsentence, metrics["bfdelsentence"] = m1.evaluate(
                masking_scheme="bfdelsentence", ds=ds_bfdelsentence, a2_col=None
            )

            ds_got_worse_with_bfdelsentence = ds_bfdelsentence.filter(
                lambda x: x["m1_bfdelsentence_None_f1"] < x["m1_supporting_None_f1"]
            )

            # distracting
            ds_got_right_with_supporting = ds.filter(
                lambda x: x["m1_supporting_None_f1"] > 0.0
            )

            ds_bfaddsentence = bf_add_sentences(ds_got_right_with_supporting)

            ds_bfaddsentence, metrics["bfaddsentence"] = m1.evaluate(
                masking_scheme="bfaddsentence",
                ds=ds_bfaddsentence,
                a2_col=None,
            )

            ds_got_worse_with_bf_add_sentence = ds_bfaddsentence.filter(
                lambda x: x["m1_bfaddsentence_None_f1"] < x["m1_supporting_None_f1"]
            )

            # reduce masked dataset to at most n=3 examples of each `id`
            # also drop examples where the delta between baseline and masked or distracted is less than adversarial_drop_thresh
            ds_got_worse_with_bf_add_sentence = reduce_to_n(
                ds_got_worse_with_bf_add_sentence,
                3,
                baseline_f1_col_name="m1_supporting_None_f1",
                exp_f1_col_name="m1_bfaddsentence_None_f1",
                adversarial_drop_thresh=adversarial_drop_thresh,
            )
            ds_got_worse_with_bfdelsentence = reduce_to_n(
                ds_got_worse_with_bfdelsentence,
                3,
                baseline_f1_col_name="m1_supporting_None_f1",
                exp_f1_col_name="m1_bfdelsentence_None_f1",
                adversarial_drop_thresh=adversarial_drop_thresh,
            )

            output_ds = combine_adversarial_ds(
                ds_got_worse_with_bf_add_sentence, ds_got_worse_with_bfdelsentence
            )

            print

        ds, metrics[masking_scheme] = m1.evaluate(
            masking_scheme=masking_scheme, ds=ds, a2_col=None
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

# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
from datasets import load_from_disk, Dataset
from oracles import T5_Bool_Oracle
from primary_models import get_m1
from secondary_model import (
    Repeater_Secondary_Model,
    OpenAI_Secondary_Model,
    Gt_Secondary_Model,
)
from dataset_utils import bf_filtering, combine_adversarial_ds
from datetime import datetime

# from masking import bf_del_sentences, bf_add_sentences, reduce_to_n
from masking import adversarial_dataset
import pandas as pd


@click.command()
@click.option("--pt_dataset_path", help="path to primary task dataset")
@click.option("--m1_path", help="path to primary model")
@click.option("--m1_arch", help="primary model architecture")
@click.option("--m2_arch", help="secondary model architecture")
@click.option("--oracle_arch", help="oracle architecture")
@click.option("--pm_eval_batch_size", help="batch size for eval", type=int)
@click.option("--oracle_eval_batch_size", help="batch size for eval", type=int)
@click.option("--masking_scheme", help="{randomsentence | bfdelsentence | None")
@click.option(
    "--adversarial_drop_thresh",
    default=0.5,
    help="include only examples in the adversarially generated examples where the delta between baseline and masked or distracted is greater than this threshold",
)
@click.option(
    "--max_adversarial_examples",
    default=3,
    help="create at most this many adversarial examples per example",
)
@click.option(
    "--downsample_pt_size",
    default=None,
    help="use at most this many examples in validation",
)
@click.option(
    "--ds_shift",
    default=0,
    help="Shift the dataset by this many examples before downsampling. Useful for debugging specific examples.",
)
@click.option(
    "--cached_adversarial_dataset_path",
    default=None,
    help="Path to save/load cached adversarial dataset. If included, skip adversarial dataset generation.",
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
    pm_eval_batch_size,
    oracle_eval_batch_size,
    masking_scheme,
    adversarial_drop_thresh,
    max_adversarial_examples,
    downsample_pt_size,
    ds_shift,
    cached_adversarial_dataset_path,
    results_filename,
    profile_only,
):
    if max_adversarial_examples is None:
        max_adversarial_examples = float("inf")
        print(
            "warning: failing to limit the number of adversarial examples may take a long time"
        )
    if ds_shift:
        assert (
            downsample_pt_size is not None
        ), "There is no reason to shift the dataset without downsampling"
    start = datetime.now()
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
        m1 = get_m1(m1_path, m1_arch, pm_eval_batch_size)
        # Receive and prepare the primary task
        metrics = {}

        if cached_adversarial_dataset_path is None:
            raw_dataset = load_from_disk(pt_dataset_path)
            if str(downsample_pt_size) != "None":
                raw_dataset = raw_dataset.select(
                    range(ds_shift, ds_shift + int(downsample_pt_size))
                )
            original_raw_dataset_len = len(raw_dataset)
            ds = raw_dataset
            # Evaluate the primary model

            # first pass
            print("m1 first pass...")
            ds, metrics["supporting"] = m1.evaluate(
                masking_scheme="supporting", ds=ds, a2_col=None
            )
            print("generating adversarial data...")
            # select and mask examples where the primary
            if masking_scheme == "bfsentence":
                ds = adversarial_dataset(
                    ds,
                    m1,
                    masking_scheme,
                    adversarial_drop_thresh,
                    max_adversarial_examples,
                )

        else:
            # Load dataset from cache
            cached_adv_df = pd.read_hdf(cached_adversarial_dataset_path)
            ds = Dataset.from_pandas(cached_adv_df)
            # Drop columns pertaining to the previous M2, which are created after this point
            drop_cols = [
                "__index_level_0__",
                "q2_bfsentence",
                "a2_bfsentence",
                "a2_is_correct_bfsentence",
                "prepped_bfsentence_a2",
                "m1_bfsentence_a2_gen",
                "m1_bfsentence_a2_f1",
                "m1_bfsentence_a2_em",
            ]  # needs fixing
            for col in drop_cols:
                if col in ds.column_names:
                    ds = ds.remove_columns(drop_cols)

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
        oracle = T5_Bool_Oracle(
            model_name=oracle_arch, batch_size=oracle_eval_batch_size
        )
        # Answer questions with the oracle
        ds = oracle.process(ds, q2_masking_scheme=masking_scheme)
        oracle.model.cpu()
        m1.model.cuda()
        ds, metrics["answered"] = m1.evaluate(
            masking_scheme=masking_scheme, ds=ds, a2_col="a2"
        )

        # Analysis
        df = pd.DataFrame(ds)
        print(f"runtime: {datetime.now()-start}")
        df.to_hdf(f"analysis_dataset_{len(raw_dataset)}_{m1_arch}_{oracle_arch}.hd5", "ds")
        # percent_oracle_correct = df[f"a2_is_correct_{masking_scheme}"].mean()
        # # print(metrics)
        # drop_cols = [
        #     "supporting_"
        # ]

        # df.to_csv(f"analysis_dataset_{len(raw_dataset)}_{m1_arch}.csv")


#         f.write(
#             f"""Model: {m1_path if m1_path else m1_arch}
# Masking Scheme:  {masking_scheme}
# Oracle:          {oracle.model_name}
# Datetime:        {now}
# Data:            {pt_dataset_path} {original_raw_dataset_len}/{len(raw_dataset)}
# Masking:         {masking_scheme}
# F1 delta:        {metrics["answered"]["f1"]-metrics[masking_scheme]["f1"]}
# Precision delta: {metrics["answered"]["precision"]-metrics[masking_scheme]["precision"]}
# Recall delta:    {metrics["answered"]["recall"]-metrics[masking_scheme]["recall"]}
# Oracle acc:      {percent_oracle_correct}
# \n
# """
#         )


if __name__ == "__main__":
    main()

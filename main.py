# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
import torch
from datasets import load_from_disk, Dataset
from oracles import *  # T5_Bool_Oracle, Bloom_Bool_Oracle, Dolly_Bool_Oracle
from primary_models import get_m1
from secondary_model import (
    Repeater_Secondary_Model,
    OpenAI_Secondary_Model,
    Gt_Secondary_Model,
)
from dataset_utils import bf_filtering, combine_adversarial_ds
from datetime import datetime
from time import sleep
import numpy as np

# from masking import bf_del_sentences, bf_add_sentences, reduce_to_n
from masking import (
    adversarial_dataset,
    randsentence_dataset,
    randdist_dataset,
    retroactively_add_distractors,
)
from pathlib import PurePath
import pandas as pd
import os

np.random.seed(42)

@click.command()
@click.option("--pt_dataset_path", help="path to primary task dataset")
@click.option("--m1_path", help="path to primary model")
@click.option("--m1_arch", help="primary model architecture")
@click.option("--m2_arch", help="secondary model architecture")
@click.option(
    "--template_id",
    help="Which prompt template to use for the secondary model. {p1, p2, p3, p4, p5, p6}",
)
@click.option("--oracle_arch", default="t5", help="oracle architecture {t5, bloom}")
@click.option(
    "--oracle_size",
    help="oracle size, t5: {small, base, large, xl, xxl}, bloom: {560m, 1b1, 1b7, 3b, 7b1}",
)
@click.option("--pm_eval_batch_size", help="batch size for eval", type=int)
@click.option("--oracle_eval_batch_size", help="batch size for eval", type=int)
@click.option("--masking_scheme", help="{randomsentence,  bfdelsentence, None")
@click.option(
    "--adversarial_drop_thresh",
    default=0.5,
    help="include only examples in the adversarially generated examples where the delta between baseline and masked or distracted is greater than this threshold",
)
@click.option(
    "--max_adversarial_examples",
    default=1,
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
@click.option(
    "--oai_cache_path",
    default=None,
    help="Path to save/load cached chatGPT responses.",
)
@click.option("--results_filename", help="path to save results")
@click.option(
    "--profile_only",
    is_flag=True,
    default=False,
    help="only profile the primary model on dataset, then exit",
)
@click.option("--save_dir", help="directory to save results to", default="results/")
def main(
    pt_dataset_path,
    m1_path,
    m1_arch,
    m2_arch,
    template_id,
    oracle_arch,
    oracle_size,
    pm_eval_batch_size,
    oracle_eval_batch_size,
    masking_scheme,
    adversarial_drop_thresh,
    max_adversarial_examples,
    downsample_pt_size,
    ds_shift,
    cached_adversarial_dataset_path,
    oai_cache_path,
    results_filename,
    profile_only,
    save_dir,
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

    # assert m2_arch in (["repeater", "openai", "gt"]) or (m2_arch.starts)
    # assert oracle_arch.startswith("t5") or oracle_arch == "dummy"

    # Evaluate the primary model
    m1 = get_m1(m1_path, m1_arch, pm_eval_batch_size)
    # Receive and prepare the primary task
    metrics = {}

    if cached_adversarial_dataset_path is None:
        ds = load_from_disk(pt_dataset_path)
        # filter out ids that don't appear in the gt dataset for speedup
        if m2_arch == "gt":
            gt_df = pd.read_csv("gt_data/non_adversarial/gt_labeled_100.csv")
            gt_ids = [x.split("_")[0] for x in gt_df["id"].tolist()]
            len_before = len(ds)
            ds = ds.filter(lambda x: x["id"].split("_")[0] in gt_ids)
            print(f"reduce to {len(ds)} / {len_before} examples")

        # downsample if a downsampling size is provided
        if str(downsample_pt_size) != "None":
            ds = ds.select(range(ds_shift, ds_shift + int(downsample_pt_size)))

        original_raw_dataset_len = len(ds)

        # first pass
        print("m1 first pass...")
        ds, metrics["supporting"] = m1.evaluate(
            masking_scheme="supporting", ds=ds, a2_col=None
        )
        # select and mask examples where the primary
        if masking_scheme == "bfsentence":
            raise NotImplementedError
            print("generating adversarial data...")
            ds = adversarial_dataset(
                ds,
                m1,
                adversarial_drop_thresh,
                max_adversarial_examples,
            )
        elif masking_scheme == "randsentence":
            do_gt = m2_arch == "gt"
            ds = randsentence_dataset(ds, m1, do_gt)
        elif masking_scheme == "randdistsentence":
            raise NotImplementedError
            ds = randdist_dataset(
                ds, m1, max_adversarial_examples
            )  # set drop thresh to -1 so no filtering happens

    else:
        # Load dataset from cache
        # assert m2_arch != "gt", "gt is not supported with cached datasets and is likely to cause errors"
        cached_adv_df = pd.read_hdf(cached_adversarial_dataset_path)
        ds = Dataset.from_pandas(cached_adv_df)
        if str(downsample_pt_size) != "None":
            ds = ds.select(range(ds_shift, ds_shift + int(downsample_pt_size)))
        # Drop columns pertaining to the previous M2, which are created after this point
        drop_cols = [
            "__index_level_0__",
            "q2_masked",
            "a2_masked",
            "a2_is_correct_masked",
            "prepped_masked_a2",
            "m1_masked_a2_gen",
            "m1_masked_a2_f1",
            "m1_masked_a2_em",
        ]  # needs fixing
        for col in drop_cols:
            if col in ds.column_names:
                ds = ds.remove_columns([col])
    # select only ground truth examples if we are doing analysis using the ground truth model
    if m2_arch == "gt":
        # gt_df = pd.read_csv("q2_gt_dataset.csv")
        # gt_qs = set(gt_df["prepped_bfdelsentence_None"].tolist())
        gt_masked_sentences = set(gt_df["masked_sentence"].tolist())
        before_len = len(ds)
        # filter based on the question cuz i forgot to include the full id in the labeling doc
        # and it wouldn't be ideal anyway cuz of suffix inconsistency
        # ds = ds.filter(lambda example: example["prepped_masked_None"] in gt_qs)
        ds = ds.filter(lambda x: x["masked_sentence"] in gt_masked_sentences)
        # select a random set of questions with one distractor added that match the annotated examples' masked sentences
        # df = ds.to_pandas()
        # pd.concat(list(x[1].head(1) for x in df.groupby("masked_sentence")))
        # dist_ds = retroactively_add_distractors(ds)
        print(len(set(ds["masked_sentence"])))
        # print(len(set(ds["distractor_sentence"])))
        print(len(set(ds["id"])))
        # get only the first example of each masked sentence
        used = set()
        relevant_examples = []
        for i, x in tqdm(enumerate(ds)):
            if x["masked_sentence"] not in used:
                relevant_examples.append(x)
                used.add(x["masked_sentence"])
        ds = Dataset.from_pandas(pd.DataFrame(data=relevant_examples))
        print
    # Create the secondary model
    if m2_arch == "repeater":
        m2 = Repeater_Secondary_Model()
    elif m2_arch == "openai":
        m2 = OpenAI_Secondary_Model(oai_cache_path, template_id)
    elif m2_arch == "gt":
        m2 = Gt_Secondary_Model(gt_df)
    else:
        raise NotImplementedError(f"m2_arch {m2_arch} not implemented")
    # Apply the secondary model
    print("m2...")
    ds = m2.process(
        ds,
        q1_col="q1",
        masking_scheme="masked",
    )
    # Save memory by moving m1 to CPU
    m1.model.cpu()
    # del m1
    torch.cuda.empty_cache()  # free up memory
    # Create the oracle
    oracle_name = f"{oracle_arch}-{oracle_size}"
    Oracle = {
        "t5": T5_Bool_Oracle,
        # "bloom": Bloom_Bool_Oracle,
        # "dolly": Dolly_Bool_Oracle,
    }[oracle_arch]
    oracle = Oracle(model_size=oracle_size, batch_size=oracle_eval_batch_size)
    # Answer questions with the oracle
    print("oracle...")
    ds = oracle.process(ds, q2_masking_scheme="masked")
    oracle.model.cpu()
    torch.cuda.empty_cache()  # free up memory

    # Bring back the primary model
    m1 = get_m1(m1_path, m1_arch, pm_eval_batch_size)
    print("m1 second pass...")
    ds, metrics["answered"] = m1.evaluate(masking_scheme="masked", ds=ds, a2_col="a2")

    # Analysis
    df = pd.DataFrame(ds)
    print(f"runtime: {datetime.now()-start}")

    # make the dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(
        save_dir,
        f"analysis_dataset_{'full' if downsample_pt_size is None else downsample_pt_size}_{masking_scheme}_{m1_arch}_{m2_arch}_{oracle_arch}_{oracle_size}_{template_id}.hd5",
    )
    desrcribe_path = PurePath(save_path+"_"+str(datetime.now())).with_suffix(".csv")
    describe_df = df[["m1_supporting_None_f1", "m1_masked_None_f1", "m1_masked_a2_f1", "m1_supporting_None_em", "m1_masked_None_em", "m1_masked_a2_em", "a2_is_correct_masked"]].astype(float).describe()
    describe_df.to_csv(desrcribe_path)

    # df.to_hdf(save_path, "ds")
    # print(f"dataset saved to {save_path}")
    # percent_oracle_correct = df[f"a2_is_correct_{masking_scheme}"].mean()
    # # print(metrics)
    # drop_cols = [
    #     "supporting_"
    # ]

    # df.to_csv(f"analysis_dataset_{len(raw_dataset)}_{m1_arch}.csv")


if __name__ == "__main__":
    main()

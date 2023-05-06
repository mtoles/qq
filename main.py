# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
import torch
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
from time import sleep
import time
import threading
from datasets import concatenate_datasets

# from masking import bf_del_sentences, bf_add_sentences, reduce_to_n
from masking import adversarial_dataset
import pandas as pd
import re


@click.command()
@click.option("--pt_dataset_path", help="path to primary task dataset")
@click.option("--m1_path", help="path to primary model")
@click.option("--m1_arch", help="primary model architecture")
@click.option("--m2_arch", help="secondary model architecture")
@click.option(
    "--template_id",
    help="Which prompt template to use for the secondary model. {p1, p2, p3, p4, p5, p6}",
)
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
@click.option("--gpus", help="GPUs to use for M1")
def main(
    pt_dataset_path,
    m1_path,
    m1_arch,
    m2_arch,
    template_id,
    oracle_arch,
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
    gpus,
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
        
        # Get GPU info
        gpu_list = get_gpus_param(gpus)
        if gpu_list == None:
            gpu_list = [0]
        
        gpu_ids = []
        for gpu in gpu_list:
            if check_cuda_device(gpu):
                gpu_ids.append(gpu)
        print(f"Valid gpu ids: {gpu_ids}")

        masking_str = f"fc_{masking_scheme}"
        
        m1_list = []
        for id in gpu_ids:
            m1_list.append(get_m1(m1_path, m1_arch, pm_eval_batch_size, f"cuda:{id}"))
        
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
            m1_1_start = time.perf_counter()
            
            if len(m1_list) == 1:
                ds, metrics["supporting"] = m1_list[0].evaluate(
                    masking_scheme="supporting", ds=ds, a2_col=None
                )
            else:
                ds, metrics["supporting"] = run_m1_multi_gpus(ds, m1_list, "supporting", None)
            
            m1_1_end = time.perf_counter()
            print("m1 first pass time: ", m1_1_end-m1_1_start)
            
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
            if str(downsample_pt_size) != "None":
                ds = ds.select(range(ds_shift, ds_shift + int(downsample_pt_size)))
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
            m2 = OpenAI_Secondary_Model(oai_cache_path, template_id)
        elif m2_arch == "gt":
            m2 = Gt_Secondary_Model()
        else:
            raise NotImplementedError(f"m2_arch {m2_arch} not implemented")
        # Apply the secondary model
        print("m2...")
        ds = m2.process(
            ds,
            q1_col="q1",
            masking_scheme=masking_scheme,
        )
        # Save memory by moving m1 to CPU
        # m1.model.cpu()
        cleanup_m1(m1_list)
        torch.cuda.empty_cache()  # free up memory
        # Create the oracle
        oracle = T5_Bool_Oracle(
            model_name=oracle_arch, batch_size=oracle_eval_batch_size
        )
        # Answer questions with the oracle
        print("oracle...")
        ds = oracle.process(ds, q2_masking_scheme=masking_scheme)
        del oracle
        torch.cuda.empty_cache()  # free up memory
        print("sleeping...")
        sleep(30)  # wait for memory to be freed

        # Bring back the primary model
        m1_list = []
        for id in gpu_ids:
            m1_list.append(get_m1(m1_path, m1_arch, pm_eval_batch_size, f"cuda:{id}"))
        print("m1 second pass...")
        m1_2_start = time.perf_counter()
        if len(m1_list) == 1:
            ds, metrics["answered"] = m1_list[0].evaluate(
                masking_scheme=masking_scheme, ds=ds, a2_col=None
            )
        else:
            ds, metrics["answered"] = run_m1_multi_gpus(ds, m1_list, masking_scheme,"a2")

        m1_2_end = time.perf_counter()
        print("m1 second pass time: ", m1_2_end-m1_2_start)

        # Analysis
        df = pd.DataFrame(ds)
        print(f"runtime: {datetime.now()-start}")
        df.to_hdf(
            f"analysis_dataset_{'full' if downsample_pt_size is None else downsample_pt_size}_{m1_arch}_{m2_arch}_{template_id}.hd5",
            "ds",
        )
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

def cleanup_m1(m1_list):
    for m1 in m1_list:
        del m1

def check_cuda_device(device_idx):
    """
    Check if a specific CUDA device exists on the system.

    Args:
    - device_idx (int): The index of the device to check.

    Returns:
    - (bool): True if the device exists, False otherwise.
    """
    device_name = f"cuda:{device_idx}"

    if not torch.cuda.is_available():
        return False
    num_devices = torch.cuda.device_count()
    if int(device_idx) >= num_devices:
        return False
    if torch.cuda.get_device_name(device_name) != '':
        return True
    print(f"CUDA device {device_name} is invalid")
    return False

def get_gpus_param(gpus):
    if re.match(r'(\d+)', gpus):
        print("Only taking the first 2 GPUs ...")
        gpu_list = re.findall(r'\d{1,2}', gpus)[:2]
        return gpu_list
    return None

# Only support 2 for now, can be modified to support 2^n GPUs where n >= 0
def run_m1_multi_gpus(ds, m1s, metric_key, a2_col):
    # Split the dataset into 2 subsets
    size0 = int(len(ds) * 0.5)
    ds_t = ds.train_test_split(test_size=size0, shuffle=True, seed=42)

    ds_map = {}
    metrics_map = {}
    ds_labels = ["train", "test"]
    threads = []
    
    for idx, m1 in enumerate(m1s):
        threads.append(threading.Thread(target=run_m1_help, args=(m1, ds_t[ds_labels[idx]], a2_col, metric_key, ds_map, metrics_map, idx)))

    # Start both threads
    for t in threads:
        t.start()

    # Wait for both threads to finish
    for t in threads:
        t.join()

    ds = concatenate_datasets([ds_map[0], ds_map[1]])
    metrics = {}
    metrics[metric_key] = metrics_map[0]
    metrics[metric_key].update(metrics_map[1])

    return ds, metrics

def run_m1_help(m1, ds, a2_col, masking_scheme, ds_map, metrics_map, idx):
    ds_map[idx], metrics_map[idx] = m1.evaluate(masking_scheme=masking_scheme, ds=ds, a2_col=a2_col)
#   print('GS type of metrics_map[idx] :', type(metrics_map[idx]))

if __name__ == "__main__":
    main()

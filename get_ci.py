import pandas as pd
import click
import os
import numpy as np


def recovery(x_a, x_i, x_c):
    if x_c == x_i:
        return None
    return (x_a - x_i) / (x_c - x_i)


def confidence_interval(series):
    clean_series = series.dropna()
    return 1.96 * clean_series.std() / np.sqrt(clean_series.count())


def confidence_interval_binary(series):
    p_hat = series.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / series.count())
    return 1.96 * se


@click.command()
@click.option(
    "--json_location",
    help="Path or dir to cached dataset file generated at end of main.py",
)
def main(json_location):
    if os.path.isdir(json_location):
        for root, dirs, files in os.walk(json_location):
            json_ds_paths = [os.path.join(root, f) for f in files if f.endswith(".json")]
    else:
        json_ds_paths = [json_location]
    for json_ds_path in json_ds_paths:
        df = pd.read_json(json_ds_path, "ds")
        df["f1_recovery"] = df.apply(
            lambda x: recovery(
                x["m1_masked_a2_f1"], x["m1_masked_None_f1"], x["m1_supporting_None_f1"]
            ),
            axis=1,
        )
        df["em_recovery"] = df.apply(
            lambda x: recovery(
                x["m1_masked_a2_em"], x["m1_masked_None_em"], x["m1_supporting_None_em"]
            ),
            axis=1,
        )
        print(json_ds_path)
        print(f"f1: {confidence_interval(df['f1_recovery'])}")
        print(f"em: {confidence_interval(df['em_recovery'])}")
        print(f"mfrr: {confidence_interval_binary(df['a2_is_correct'])}")
        pass


if __name__ == "__main__":
    main()

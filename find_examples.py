import pandas as pd
import click
import os


@click.command()
@click.option(
    "--json_dir",
    help="Path to cached dataset file generated at end of main.py",  # probably the gt folder
)
def main(json_dir):
    # get a list of all files in json_dir
    if os.path.isdir(json_dir):
        for root, dirs, files in os.walk(json_dir):
            json_ds_paths = [os.path.join(root, f) for f in files]
    else:
        json_ds_paths = [json_dir]
    json_ds_paths = sorted(json_ds_paths)
    # keep only the json files
    json_ds_paths = [f for f in json_ds_paths if f.endswith(".json")]

    interesting_cols = [
        "id",  # shared
        "q1",  # shared
        "fc_masked",  # shared
        "masked_sentence",  # shared
        "a1",  # shared
        "q2_masked",  # not shared
        "a2_masked",  # not shared
        "m1_masked_None_f1",  # shared
        "m1_masked_a2_f1",  # not shared
        "a2_is_correct_masked",  # not shared
        "m1_masked_a2_gen",  # not shared
    ]

    single_dfs = []
    df_gt, df_alpaca, df_gpt35, df_gpt4, df_repeater = None, None, None, None, None
    for json_ds_path in json_ds_paths:
        # get the m2 name from the filename
        m2_name = json_ds_path.split("_")[6]

        single_df = pd.read_json(json_ds_path)
        single_df = single_df[interesting_cols]
        single_df["m2"] = m2_name

        if m2_name == "gt":
            df_gt = single_df
        elif m2_name == "alpaca":
            df_alpaca = single_df
        elif m2_name == "gpt-3.5-turbo":
            df_gpt35 = single_df
        elif m2_name == "gpt-4":
            df_gpt4 = single_df
        elif m2_name == "repeater":
            df_repeater = single_df
        else:
            raise ValueError(f"Unknown m2 name: {m2_name}")

    shared_cols = [
        "id",
        "q1",
        "fc_masked",
        "masked_sentence",
        "a1",
        "m1_masked_None_f1",
    ]
    df_shared = df_gt[shared_cols]

    # drop shared columns
    df_gt = df_gt.drop(columns=shared_cols)
    df_alpaca = df_alpaca.drop(columns=shared_cols)
    df_gpt35 = df_gpt35.drop(columns=shared_cols)
    df_gpt4 = df_gpt4.drop(columns=shared_cols)
    df_repeater = df_repeater.drop(columns=shared_cols)

    df_all = pd.concat(
        [
            df_gt.add_prefix("gt_"),
            df_alpaca.add_prefix("alpaca_"),
            df_gpt35.add_prefix("gpt35_"),
            df_gpt4.add_prefix("gpt4_"),
            # df_repeater.add_prefix("repeater_"), # ignore repeater
        ],
        axis=1,
    )

    ### look for interesting stuff ###

    # get examples where model "gt" succeeds but 4/35/alpaca fail
    df_a = df_gt[
        (df_gt.m1_masked_a2_f1 > df_gpt4.m1_masked_a2_f1)
        & (df_gt.m1_masked_a2_f1 > df_gpt35.m1_masked_a2_f1)
        & (df_gt.m1_masked_a2_f1 > df_alpaca.m1_masked_a2_f1)
    ]
    # get examples where model "gt" and "gpt4" succeed but 35/alpaca fail
    # df_a = df_gt[
    #     (
    #         (
    #             (df_gt.m1_masked_a2_f1 > df_gpt35.m1_masked_a2_f1)
    #             | (df_gt.m1_masked_a2_f1 > df_alpaca.m1_masked_a2_f1)
    #         )
    #         & ~df_gt.a2_is_correct_masked
    #     )
    #     | (
    #         (
    #             (df_gpt4.m1_masked_a2_f1 > df_gpt35.m1_masked_a2_f1)
    #             | (df_gpt4.m1_masked_a2_f1 > df_alpaca.m1_masked_a2_f1)
    #         )
    #         & (~df_gpt4.a2_is_correct_masked)
    #     )
    #     # & (df_shared.m1_masked_None_f1 < max(df_gpt4.m1_masked_a2_f1, df_gt.m1_masked_a2_f1))
    # ]

    df_output = df_all.iloc[df_a.index]
    df_output = df_output.drop(
        [
            "gt_m2",
            "alpaca_m2",
            "gpt35_m2",
            "gpt4_m2",
            "gpt4_a2_is_correct_masked",
            "gpt35_a2_is_correct_masked",
            "alpaca_a2_is_correct_masked",
            "gt_a2_is_correct_masked",
        ],
        axis=1,
    )
    ordered_col = df_output.columns.tolist()
    ordered_col.sort(key=lambda x: x[::-1])  # sort by reverse string
    df_output = df_output[ordered_col]
    df_output = df_shared.join(df_output, how="inner")
    df_output.to_csv("find_examples/find_examples.csv", index=False)
    print()


if __name__ == "__main__":
    main()

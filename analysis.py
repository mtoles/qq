import pandas as pd
import click


@click.command()
@click.option(
    "--hdf_ds_path", help="Path to cached dataset file generated at end of main.py"
)
def main(hdf_ds_path):
    df_raw = pd.read_hdf(hdf_ds_path, "ds")
    interesting_cols = [
        "prepped_bfsentence_None",
        "masked_sentence",
        "q2_bfsentence",
        "a2_bfsentence",
        "m1_bfsentence_None_f1",
        "m1_bfsentence_a2_f1",
        "a2_is_correct_bfsentence",
    ]
    df = df_raw[interesting_cols]
    df["delta_l"] = df["m1_bfsentence_None_f1"] - df["m1_bfsentence_a2_f1"] # mean: -0.422
    df["did_improve"] = (df["m1_bfsentence_None_f1"]<df["m1_bfsentence_a2_f1"]) # sum: 3438
    df["wrong_answer_but_improved"] = (df["a2_is_correct_bfsentence"]==False) & df["did_improve"] # sum: 1184

    df = df[interesting_cols]
    print
    # print(df)
    # df[df[["fc_bfsentence"]].duplicated(keep=False)].sort_values(by="fc_bfsentence")[['id', 'fc_bfsentence']]
    # numb: 7230, 152, 6496


if __name__ == "__main__":
    main()

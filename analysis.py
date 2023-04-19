import pandas as pd
import click


# a function that takes a row in a dataframe and returns true if the row exists elsewhere in the dataframe
def find_duplicates(df, row):
    return df.loc[df["name"] == row["name"]].shape[0] > 1


@click.command()
@click.option(
    "--hdf_ds_path", help="Path to cached dataset file generated at end of main.py"
)
def main(hdf_ds_path):
    df = pd.read_hdf(hdf_ds_path, "ds")
    df.duplicated
    for i in range(len(df)):
        dup_df = df[df.apply(find_duplicates, i, axis=1)]
    print

    # print(df)
    # df[df[["fc_bfsentence"]].duplicated(keep=False)].sort_values(by="fc_bfsentence")[['id', 'fc_bfsentence']]
    # numb: 7230, 152, 6496

if __name__ == "__main__":
    main()

import pandas as pd
import click

@click.command()
@click.option(
    "--hdf_ds_path", help="Path to cached dataset file generated at end of main.py"
)
def main(hdf_ds_path):
    df = pd.read_hdf(hdf_ds_path, "ds")
    print(df)
    print

    # print(df)
    # df[df[["fc_bfsentence"]].duplicated(keep=False)].sort_values(by="fc_bfsentence")[['id', 'fc_bfsentence']]
    # numb: 7230, 152, 6496

if __name__ == "__main__":
    main()

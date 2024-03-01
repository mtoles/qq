import pandas as pd
import click

@click.command()
@click.option('--ds_path', default=None, help='Path to the dataset')
def main(ds_path):
    if ds_path is None:
        # create the dataset from scratch via supervised fine tuning
        
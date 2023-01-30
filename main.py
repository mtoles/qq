# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
from datasets import load_from_disk
from oracle import Dummy_oracle
from secondary_model import Dummy_secondary_model
from primary_models import BigBird_PM
from dataset_utils import drop_unanswerable


@click.command()
@click.option("--pt_dataset_path", help="path to primary task dataset")
@click.option("--pm_paths", multiple=True, help="path to primary model")
@click.option("--pm_arch", multiple=True, help="primary model architecture")
@click.option("--masking_scheme", help="{randomsentence | None")
@click.option("--downsample_pt_size", default=None, type=int, help="use at most this many examples in validation")
def main(pt_dataset_path, pm_paths, pm_arch, masking_scheme, downsample_pt_size):
    masking_str = f"fc_{masking_scheme}"

    # Unit Tests
    for pma in pm_arch:
        assert pma in ["bigbird", "gpt_neox"]
    for model_num, pm_path in enumerate(pm_paths):
        p_model_type = pm_arch[model_num]

        # Load primary model
        if p_model_type == "bigbird":
            pm = BigBird_PM(pm_path)
        elif pm_path == "gpt_neox":
            raise NotImplementedError

        # Receive and prepare the primary task
        pt_dataset = load_from_disk(pt_dataset_path)
        if downsample_pt_size is not None:
            pt_dataset = pt_dataset.select(range(downsample_pt_size)) 
        pt_dataset = drop_unanswerable(pt_dataset, masking_scheme=masking_scheme, load_from_cache_file=True)
        pt_dataset = pm.prepare_data(pt_dataset)

        example = pt_dataset[masking_str][0]
        print(example)

        pm_output = pm(example)

        # perform secondary task
        secondary_model = Dummy_secondary_model()
        query = secondary_model(example)

        # consult the oracle
        oracle = Dummy_oracle(pm_output, corpus="This is a test. This is another test. This is a third test.")
        answer = oracle.consult(query)

        # TODO: Insert answer into the primary task
        

        pass


if __name__ == "__main__":
    main()

# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
from datasets import load_from_disk
from oracle import Dummy_oracle
from secondary_model import Dummy_secondary_model
from primary_models import BigBird_PM, GPTNeoX_PM
from dataset_utils import drop_unanswerable
from datetime import datetime


@click.command()
@click.option("--pt_dataset_path", help="path to primary task dataset")
@click.option("--pm_path", help="path to primary model")
@click.option("--pm_arch", help="primary model architecture")
@click.option("--eval_batch_size", default=2, help="batch size for eval")
@click.option("--masking_scheme", help="{randomsentence | None")
@click.option(
    "--downsample_pt_size",
    default=None,
    help="use at most this many examples in validation",
)
@click.option("--results_filename", help="path to save results")
def main(pt_dataset_path, pm_path, pm_arch, eval_batch_size, masking_scheme, downsample_pt_size, results_filename):
    now = datetime.now().strftime("y%m%d-%H%M%S")
    
    with open(f"inf_logs/{results_filename}.txt", "a") as f:

        masking_str = f"fc_{masking_scheme}"

        # Unit Tests
        assert pm_arch in ["bigbird", "gptneox"]

        # Receive and prepare the primary task
        pt_dataset = load_from_disk(pt_dataset_path)
        original_pt_dataset_size = len(pt_dataset)
        if str(downsample_pt_size) != "None":
            pt_dataset = pt_dataset.select(range(int(downsample_pt_size)))

        # Load primary model
        if pm_arch == "bigbird":
            pm = BigBird_PM(pm_path, eval_batch_size=eval_batch_size, raw_val_dataset=pt_dataset)
        elif pm_arch == "gptneox":
            pm = GPTNeoX_PM(eval_batch_size=eval_batch_size, raw_val_dataset=pt_dataset)
        else:
            raise NotImplementedError

        pm.prepare_data(masking_scheme=masking_scheme)
        eval_metrics = pm.trainer.evaluate()
        f.write(
            f"""Model:     {pm_path}
Datetime:  {now}
Data:      {pt_dataset_path} {original_pt_dataset_size}/{len(pt_dataset)}
Masking:   {masking_scheme}
F1:        {eval_metrics["eval_f1"]}
Precision: {eval_metrics["eval_precision"]}
Recall:    {eval_metrics["eval_recall"]}\n\n""")

        # example = pt_dataset[masking_str][0]
        # print(example)

        # pm_output = pm(example)

        # perform secondary task
        # secondary_model = Dummy_secondary_model()
        # query = secondary_model(example)

        # # consult the oracle
        # oracle = Dummy_oracle(pm_output, corpus="This is a test. This is another test. This is a third test.")
        # answer = oracle.consult(query)

        # TODO: Insert answer into the primary task

    pass


if __name__ == "__main__":
    main()

# Abbreviations:
#  p: primary
#  pm: primary model
#  pt: primary task

import click
from datasets import load_from_disk
from oracles import Dummy_Oracle, T5_Oracle
from secondary_model import Dummy_Secondary_Model
from primary_models import BigBird_PM, T5_PM
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
def main(
    pt_dataset_path,
    pm_path,
    pm_arch,
    eval_batch_size,
    masking_scheme,
    downsample_pt_size,
    results_filename,
):
    now = datetime.now().strftime("Y%m%d-%H%M%S")
    if results_filename is None:
        results_filename = f"{pm_arch}-{downsample_pt_size}-{masking_scheme}-{now}"

    with open(f"inf_logs/{results_filename}.txt", "a") as f:

        masking_str = f"fc_{masking_scheme}"

        # Unit Tests
        assert pm_arch in [
            "bigbird",
            "t5-small",
            "t5-base",
            "t5-large",
            "t5-xl",
            "t5-xxl",
        ]

        # Receive and prepare the primary task
        pt_dataset = load_from_disk(pt_dataset_path)
        original_pt_dataset_size = len(pt_dataset)
        if str(downsample_pt_size) != "None":
            pt_dataset = pt_dataset.select(range(int(downsample_pt_size)))

        # Load primary model
        if pm_arch == "bigbird":
            pm = BigBird_PM(
                pm_path, eval_batch_size=eval_batch_size, raw_val_dataset=pt_dataset
            )
        elif pm_arch.startswith("t5"):
            pm = T5_PM(
                eval_batch_size=eval_batch_size,
                raw_val_dataset=pt_dataset,
                model_name=pm_arch,
            )
        else:
            raise NotImplementedError

        pm.prepare_data(masking_scheme=masking_scheme)

        sm = Dummy_Secondary_Model()

        pm.prepped_val_dataset = sm.process(
            pm.prepped_val_dataset,
            primary_question_col="question",
            context_col=masking_str,
        )

        oracle = T5_Oracle(model_name="t5-small")

        pm.prepped_val_dataset = oracle.process(
            pm.prepped_val_dataset, secondary_question_col="secondary_question"
        )
        # oracle.forward(
        #     pm.prepped_val_dataset[0], secondary_question_col="secondary_question"
        # )

        # eval_metrics = pm.evaluate(masking_scheme="randomsentence")
        none_metrics = pm.evaluate(masking_scheme="supporting")
        print(none_metrics)
        rs_metrics = pm.evaluate(masking_scheme="randomsentence")
        print(rs_metrics)


        f.write(
            f"""Model:     {pm_path if pm_path else pm_arch}
Datetime:  {now}
Data:      {pt_dataset_path} {original_pt_dataset_size}/{len(pt_dataset)}
Masking:   {masking_scheme}
F1:        {eval_metrics["f1"]}
Precision: {eval_metrics["precision"]}
Recall:    {eval_metrics["recall"]}\n\n"""
        )

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

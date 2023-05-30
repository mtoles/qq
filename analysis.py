import pandas as pd
import click
import os


pd.options.mode.chained_assignment = None  # default='warn'


@click.command()
@click.option(
    "--hdfs_dir",
    help="Path to cached dataset file generated at end of main.py",
)
def main(hdfs_dir):
    # get a list of all files in hdfs_dir
    for root, dirs, files in os.walk(hdfs_dir):
        hdf_ds_paths = [os.path.join(root, f) for f in files]

        # old code for finding the best prompt
        # delta_l = []
        # find the prompt with the best delta_l
        # for i, hdf_ds_path in enumerate(hdf_ds_paths):
        #     df_raw = pd.read_hdf(hdf_ds_path, "ds")

        #     df_raw["delta_l"] = (
        #         df_raw["m1_bfsentence_a2_f1"] - df_raw["m1_bfsentence_None_f1"]
        #     )  # mean: -0.422

        #     delta_l.append(df_raw["delta_l"].mean())
        # clen up the prompt with the best delta_l
        # best_delta_l = max(delta_l)
        # i_best_delta_l = delta_l.index(best_delta_l)
        # print(f"Best delta_l: {best_delta_l} at index {i_best_delta_l}")
        # df = pd.read_hdf(hdf_ds_paths[i_best_delta_l], "ds")
    # sort them
    hdf_ds_paths = sorted(hdf_ds_paths)
    for hdf_ds_path in hdf_ds_paths:
        df_raw = pd.read_hdf(hdf_ds_path, "ds")
        interesting_cols = [
            "id",
            "prepped_bfsentence_None",
            "masked_sentence",
            "a1",
            "q2_bfsentence",
            "a2_bfsentence",
            "m1_bfsentence_None_f1",
            "m1_bfsentence_a2_f1",
            "a2_is_correct_bfsentence",
            "m1_bfsentence_a2_gen",
        ]
        df = df_raw[interesting_cols]
        df["did_improve"] = df["m1_bfsentence_None_f1"] < df["m1_bfsentence_a2_f1"]
        df["wrong_answer_but_improved"] = (
            df["a2_is_correct_bfsentence"] == False
        ) & df["did_improve"]
        df["delta_l"] = df["m1_bfsentence_a2_f1"] - df["m1_bfsentence_None_f1"]
        # separate id from add/delete type
        df["type"] = df["id"].apply(lambda x: x.split("_")[1][0])
        df["id"] = df["id"].apply(lambda x: x.split("_")[0])
        df["a2_is_masked_sentence"] = df.apply(
            lambda x: x["masked_sentence"] in x["a2_bfsentence"], axis=1
        )

        num_questions = len(df)
        num_a2_is_masked_sentence = sum(df["a2_is_masked_sentence"])
        num_a2_is_distractor = num_questions - num_a2_is_masked_sentence
        num_masked_sentence_improved = sum(
            df["a2_is_masked_sentence"] & df["did_improve"]
        )
        num_masked_sentence_not_improved = sum(
            df["a2_is_masked_sentence"] & ~df["did_improve"]
        )
        num_distractor_improved = sum(~df["a2_is_masked_sentence"] & df["did_improve"])
        num_distractor_not_improved = sum(
            ~df["a2_is_masked_sentence"] & ~df["did_improve"]
        )
        # print(f"num_questions: {num_questions}")
        # print(f"num_a2_is_masked_sentence: {num_a2_is_masked_sentence}")
        # print(f"num_a2_is_distractor: {num_a2_is_distractor}")
        # print(f"num_masked_sentence_improved: {num_masked_sentence_improved}")
        # print(f"num_masked_sentence_not_improved: {num_masked_sentence_not_improved}")
        # print(f"num_distractor_improved: {num_distractor_improved}")
        # print(f"num_distractor_not_improved: {num_distractor_not_improved}")
        print(hdf_ds_path)
        print(f"delta l: {df['delta_l'].mean()}")
        # print(f"percent improved: {sum(df['did_improve']) / len(df)}")
        print


if __name__ == "__main__":
    main()

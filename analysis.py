import pandas as pd
import click
import os


pd.options.mode.chained_assignment = None  # default='warn'


@click.command()
@click.option(
    "--hdfs_dir",
    help="Path to cached dataset file generated at end of main.py",
)
@click.option(
    "--gt_only",
    flag_value=True,
    help="Filter out all examples not present in ground truth"
)
def main(hdfs_dir, gt_only):

 
    # get a list of all files in hdfs_dir
    if os.path.isdir(hdfs_dir):
        for root, dirs, files in os.walk(hdfs_dir):
            hdf_ds_paths = [os.path.join(root, f) for f in files]
    else:
        hdf_ds_paths = [hdfs_dir]

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
            "prepped_masked_None",
            "masked_sentence",
            "a1",
            "q2_masked",
            "a2_masked",
            "m1_masked_None_f1",
            "m1_masked_a2_f1",
            "a2_is_correct_masked",
            "m1_masked_a2_gen",
        ]
        # df = df_raw[interesting_cols]
        df = df_raw

        if gt_only:
            # filter out all examples not present in ground truth
            gt_df = pd.read_csv("gt_data/non_adversarial/gt_labeled_100.csv")
            gt_ids = gt_df["id"].tolist()
            len_before = len(df)
            gt_masked_sentences = set(gt_df["masked_sentence"].tolist())
            df = df[df.apply(lambda x: x["masked_sentence"] in gt_masked_sentences, axis=1)]

            # filter based on the question cuz i forgot to include the full id in the labeling doc
            # and it wouldn't be ideal anyway cuz of suffix inconsistency
            # ds = ds.filter(lambda example: example["prepped_masked_None"] in gt_qs)
            # df = df[df["masked_sentence"] in gt_masked_sentences]

        # rename randsentence col to masked col

        df["did_improve"] = df["m1_masked_None_f1"] < df["m1_masked_a2_f1"]
        df["got_worse"] = df["m1_masked_None_f1"] > df["m1_masked_a2_f1"]
        df["stayed_same"] = df["m1_masked_None_f1"] == df["m1_masked_a2_f1"]
        df["wrong_answer_but_improved"] = (df["a2_is_correct_masked"] == False) & df[
            "did_improve"
        ]
        df["delta_l"] = df["m1_masked_a2_f1"] - df["m1_masked_None_f1"]
        # separate id from add/delete type
        # df["type"] = df["id"].apply(lambda x: x.split("_")[1][0])
        df["id"] = df["id"].apply(lambda x: x.split("_")[0])
        df["a2_is_masked_sentence"] = df.apply(
            lambda x: x["masked_sentence"] in x["a2_masked"], axis=1
        )
        # df["a2_type"] = df.apply(get_answer_type, axis=1)
        df["a2_in_a1"] = df.apply(lambda x: x["a2_masked"] in x["a1"], axis=1)
        num_questions = len(df)
        num_a2_is_masked_sentence = sum(df["a2_is_masked_sentence"])
        num_a2_is_distractor = num_questions - num_a2_is_masked_sentence
        num_masked_sentence_improved = sum(
            df["a2_is_masked_sentence"] & df["did_improve"]
        )
        num_masked_sentence_stayed_same = sum(
            df["a2_is_masked_sentence"] & df["stayed_same"]
        )
        num_masked_sentence_got_worse = sum(
            df["a2_is_masked_sentence"] & df["got_worse"]
        )
        num_distractor_improved = sum(~df["a2_is_masked_sentence"] & df["did_improve"])
        num_distractor_stayed_same = sum(
            ~df["a2_is_masked_sentence"] & df["stayed_same"]
        )
        num_distractor_got_worse = sum(~df["a2_is_masked_sentence"] & df["got_worse"])
        print(f"num_questions: {num_questions}")
        print(f"num_a2_is_masked_sentence: {num_a2_is_masked_sentence}")
        print(f"num_a2_is_distractor: {num_a2_is_distractor}")
        print(f"num_masked_sentence_improved: {num_masked_sentence_improved}")
        print(f"num_masked_sentence_stayed_same: {num_masked_sentence_stayed_same}")
        print(f"num_masked_sentence_got_worse: {num_masked_sentence_got_worse}")
        print(f"num_distractor_improved: {num_distractor_improved}")
        print(f"num_distractor_stayed_same: {num_distractor_stayed_same}")
        print(f"num_distractor_got_worse: {num_distractor_got_worse}")
        print(hdf_ds_path)
        print(f"delta l: {df['delta_l'].mean()}")
        # print(f"percent improved: {sum(df['did_improve']) / len(df)}")
        print

        # reorder columns for saving df
        first_cols = ["id", "q1", "a1", "masked_sentence", "fc_masked"]
        column_order = first_cols + [col for col in df.columns if col not in first_cols]
        df = df[column_order]
        # sample the df for gt annotation
        # save the df
        print


def get_answer_type(example):
    supporting = example["context_supporting"]["sentences"]
    # flatten the supporting list of lists
    supporting = [item for sublist in supporting for item in sublist]
    distractor = example["context_distractor"]["sentences"]
    distractor = [item for sublist in distractor for item in sublist]
    if example["a2_is_correct_masked"]:
        return "correct"
    elif example["a2_masked"] in supporting:
        return "supporting"
    elif example["a2_masked"] in distractor:
        return "distractor"
    else:
        raise ValueError("a2_masked not found in supporting or distractor")


if __name__ == "__main__":
    main()

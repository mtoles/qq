import pandas as pd
import click


@click.command()
@click.option(
    "--hdf_ds_paths",
    help="Path to cached dataset file generated at end of main.py",
    multiple=True,
)
def main(hdf_ds_paths):
    delta_l = []
    # find the prompt with the best delta_l
    for i, hdf_ds_path in enumerate(hdf_ds_paths):
        df_raw = pd.read_hdf(hdf_ds_path, "ds")

        df_raw["delta_l"] = (
            df_raw["m1_bfsentence_a2_f1"] - df_raw["m1_bfsentence_None_f1"]
        )  # mean: -0.422

        delta_l.append(df_raw["delta_l"].mean())
    # clen up the prompt with the best delta_l
    best_delta_l = max(delta_l)
    i_best_delta_l = delta_l.index(best_delta_l)
    print(f"Best delta_l: {best_delta_l} at index {i_best_delta_l}")
    df = pd.read_hdf(hdf_ds_paths[i_best_delta_l], "ds")
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
        "m1_bfsentence_a2_gen"
    ]
    df = df_raw[interesting_cols]
    df["did_improve"] = df["m1_bfsentence_None_f1"] < df["m1_bfsentence_a2_f1"]
    df["wrong_answer_but_improved"] = (df["a2_is_correct_bfsentence"] == False) & df[
        "did_improve"
    ]
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
    num_masked_sentence_improved = sum(df["a2_is_masked_sentence"] & df["did_improve"])
    num_masked_sentence_not_improved = sum(
        df["a2_is_masked_sentence"] & ~df["did_improve"]
    )
    num_distractor_improved = sum(~df["a2_is_masked_sentence"] & df["did_improve"])
    num_distractor_not_improved = sum(~df["a2_is_masked_sentence"] & ~df["did_improve"])
    print(f"num_questions: {num_questions}")
    print(f"num_a2_is_masked_sentence: {num_a2_is_masked_sentence}")
    print(f"num_a2_is_distractor: {num_a2_is_distractor}")
    print(f"num_masked_sentence_improved: {num_masked_sentence_improved}")
    print(f"num_masked_sentence_not_improved: {num_masked_sentence_not_improved}")
    print(f"num_distractor_improved: {num_distractor_improved}")
    print(f"num_distractor_not_improved: {num_distractor_not_improved}")
    print(f"delta l: {df['delta_l'].mean()}")
    print(f"percent improved: {sum(df['did_improve']) / len(df)}")

    # # display examples where the model improved despite using a distractor
    # df_distractor_improved = df[df["did_improve"] & ~df["a2_is_masked_sentence"]]
    # for i in range(len(df_distractor_improved)):
    #     print(i)
    #     print(df_distractor_improved.iloc[i]["prepped_bfsentence_None"])
    #     print(df_distractor_improved.iloc[i]["masked_sentence"])
    #     print(df_distractor_improved.iloc[i]["a2_bfsentence"])
    # print

    # display examples where the model did not improve despite using a masked sentence
    # df_masked_sentence_not_improved = df[
    #     ~df["did_improve"] & df["a2_is_masked_sentence"]
    # ]
    # for i in range(len(df_masked_sentence_not_improved)):
    #     print(i)
    #     print(df_masked_sentence_not_improved.iloc[i]["prepped_bfsentence_None"])
    #     print(df_masked_sentence_not_improved.iloc[i]["masked_sentence"])
    #     print(df_masked_sentence_not_improved.iloc[i]["a2_bfsentence"])
    # print

    # create a ground truth dataset for ground truth annotation
    # get first 100 rows with unique ids
    # gt_selection_df = df[df["id"].duplicated(keep=False) == False].iloc[:100][
    #     ["id", "prepped_bfsentence_None", "masked_sentence", "a1"]
    # ]
    # gt_selection_df["prepped_bfsentence_None"] = gt_selection_df[
    #     "prepped_bfsentence_None"
    # ].apply(lambda x: x[:-38])
    # gt_selection_df.to_csv("q2_gt_adv_dataset.csv", index=False)

    # read in ground truth
    # gt_df = pd.read_csv("q2_gt_dataset.csv")
    # gt_df = gt_df[gt_df["q2_gt"].isna() == False][["id", "q2_gt"]]

    # df_ids = set(df["id"])
    # gt_df["in_df"] = gt_df["id"].apply(lambda x: x in df_ids)
    # print(len(set(df["id"]).intersection(set(gt_df["id"]))))
    # df = df[interesting_cols]
    print()
    # print(df)
    # df[df[["fc_bfsentence"]].duplicated(keep=False)].sort_values(by="fc_bfsentence")[['id', 'fc_bfsentence']]
    # numb: 7230, 152, 6496


if __name__ == "__main__":
    main()

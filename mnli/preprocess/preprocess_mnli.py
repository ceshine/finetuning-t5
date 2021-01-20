from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

label_encode_dict = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}


def encode_label(df):
    df["label"] = df["gold_label"].apply(lambda x: label_encode_dict[x])


if __name__ == "__main__":
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    df_train = pd.read_csv("data/multinli_1.0/multinli_1.0_train.txt", sep="\t", error_bad_lines=False)
    df_test_matched = pd.read_csv("data/multinli_1.0/multinli_1.0_dev_matched.txt", sep="\t")
    df_test_mismatched = pd.read_csv("data/multinli_1.0/multinli_1.0_dev_mismatched.txt", sep="\t")
    all_dfs = [df_train, df_test_matched, df_test_mismatched]
    for i, df in enumerate(all_dfs):
        # print(df["gold_label"].unique())
        print("Filtering out problematic rows...")
        print("Before:", df.shape[0])
        df = df[(df["gold_label"] != "-") & (~df["sentence1"].isnull()) & (~df["sentence2"].isnull())].copy()
        print("After:", df.shape[0])
        print("=" * 20)
        df["label"] = df["gold_label"].apply(lambda x: label_encode_dict[x])
        df["premise"] = df["sentence1"]
        df["hypothesis"] = df["sentence2"]
        df["id"] = df["pairID"]
        all_dfs[i] = df[["id", "premise", "hypothesis", "label"]].copy()
        print("premise stats")
        print(df["premise"].str.len().describe())
        print(all_dfs[i].shape)
    df_train, df_test_matched, df_test_mismatched = all_dfs
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, valid_index in sss.split(df_train, df_train["label"]):
        df_valid = df_train.iloc[valid_index].reset_index(drop=True)
        df_train = df_train.iloc[train_index].reset_index(drop=True)
        print(df_train.shape, df_valid.shape)
    Path("data/multinli").mkdir(exist_ok=True)
    df_train.to_csv('data/multinli/train_split.csv', index=False)
    df_valid.to_csv('data/multinli/valid.csv', index=False)
    df_test_matched.to_csv('data/multinli/test_matched.csv', index=False)
    df_test_mismatched.to_csv('data/multinli/test_mismatched.csv', index=False)

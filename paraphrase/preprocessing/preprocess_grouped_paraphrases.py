import re
from pathlib import Path

import typer
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize.treebank import TreebankWordDetokenizer

from tokenize_dataset import DATA_PATH


def main(data_folder: str, detokenize: bool = False, val_ratio: float = 0.1):
    df_groups = pd.read_csv(Path(data_folder) / "phrase_groups.csv")
    phrases = []
    detokenizer = TreebankWordDetokenizer()
    with open(Path(data_folder) / "phrases.txt") as fin:
        for line in fin.readlines():
            if detokenize:
                line = detokenizer.detokenize(line.split(" "))
            line = re.sub(r" +", " ", line)
            phrases.append(line.strip())
    groups = df_groups.groupby('paraphrase_group_index')['phrase_index'].apply(list).values
    print(pd.Series([len(x) for x in groups]).value_counts())
    buffer = []
    for group in groups:
        for i in range(len(group)-1):
            for j in range(i+1, len(group)):
                buffer.append((phrases[group[i]], phrases[group[j]]))
    df = pd.DataFrame(buffer, columns=["sentence1", "sentence2"])
    name = Path(data_folder).name.split("_")[0]
    train, valid = train_test_split(df, test_size=val_ratio)
    train.to_csv(DATA_PATH / f"{name}_train.csv", index=False)
    valid.to_csv(DATA_PATH / f"{name}_valid.csv", index=False)


if __name__ == "__main__":
    typer.run(main)

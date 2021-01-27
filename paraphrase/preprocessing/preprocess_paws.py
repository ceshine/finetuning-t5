import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize.treebank import TreebankWordDetokenizer

DATA_FILES = [
    "data/paws-swap/train.tsv",
    "data/paws-final/train.tsv",
    "data/paws-final/dev.tsv",
    "data/paws-final/test.tsv",
    "data/paws-qqp/paws-qqp-train.tsv"
]

if __name__ == "__main__":
    buffer = []
    for filepath in DATA_FILES:
        df = pd.read_csv(filepath, sep="\t")
        buffer.append(df[df.label == 1][["sentence1", "sentence2"]].copy())
        print(filepath, buffer[-1].shape[0])
    df_final = pd.concat(buffer, axis=0)
    detokenizer = TreebankWordDetokenizer()
    df_final["sentence1"] = df_final["sentence1"].apply(lambda x: detokenizer.detokenize(x.split(" ")))
    df_final["sentence2"] = df_final["sentence2"].apply(lambda x: detokenizer.detokenize(x.split(" ")))
    del buffer
    train, test = train_test_split(df_final, test_size=0.2)
    valid, test = train_test_split(test, test_size=0.5)
    train.to_csv('data/paws_train.csv', index=False)
    valid.to_csv('data/paws_valid.csv', index=False)
    test.to_csv('data/paws_test.csv', index=False)

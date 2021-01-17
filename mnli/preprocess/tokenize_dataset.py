import enum
from pathlib import Path

import typer
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import MT5Tokenizer, T5Tokenizer

DATA_PATH = Path("data/")
CACHE_PATH = Path("cache/")
CACHE_PATH.mkdir(exist_ok=True, parents=True)

# label_dict = {
#     0: "entailment",
#     1: "neutral",
#     2: "contradiction"
# }


class Corpus(enum.Enum):
    KAGGLE = "kaggle"
    MNLI = "multinli"


def process_file(data: pd.DataFrame, tokenizer: MT5Tokenizer, batch_size: int):
    premise_ids = []
    hypothesis_ids = []
    if "label" in data.columns:
        assert data.label.max() == data.label.nunique() - 1
        assert data.label.min() == 0
        labels = data.label.astype("int")[:, np.newaxis]
    else:
        labels = None
    for i in tqdm(range(0, len(data), batch_size), ncols=100):
        batch = data.iloc[i:i+batch_size]
        # tokenize premise
        premise_ids.extend(
            tokenizer.batch_encode_plus(
                batch["premise"].values, return_attention_mask=False
            )["input_ids"]
        )
        # tokenize hypothesis
        hypothesis_ids.extend(
            tokenizer.batch_encode_plus(
                batch["hypothesis"].values.tolist(), return_attention_mask=False
            )["input_ids"]
        )
    return premise_ids, hypothesis_ids, labels


def main(dataset: Corpus, tokenizer_name: str = "google/mt5-small", batch_size: int = 1024):
    (CACHE_PATH / dataset.value).mkdir(parents=True, exist_ok=True)
    if "mt5" in tokenizer_name:
        tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    for datafile in ("train_split.csv", "valid.csv", "test.csv", "test_matched.csv", "test_mismatched.csv"):
        if not (DATA_PATH / dataset.value / datafile).exists():
            continue
        print(datafile)
        data = pd.read_csv(DATA_PATH / dataset.value / datafile)
        premise_ids, hypothesis_ids, labels = process_file(data, tokenizer, batch_size)
        joblib.dump([premise_ids, hypothesis_ids, labels], CACHE_PATH / dataset.value / (Path(datafile).stem + ".jbl"))
        print(CACHE_PATH / dataset.value / (Path(datafile).stem + ".jbl"))


if __name__ == "__main__":
    typer.run(main)

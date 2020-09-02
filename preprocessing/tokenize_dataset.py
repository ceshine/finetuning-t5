import enum
from pathlib import Path

import typer
import joblib
from tqdm import tqdm
import pandas as pd
from transformers import T5Tokenizer

DATA_PATH = Path("data/")
CACHE_PATH = Path("cache/")


class Dataset(enum.Enum):
    QUORA = "quora"
    PAWS = "paws"


def process_file(data: pd.DataFrame, tokenizer: T5Tokenizer, batch_size: int):
    input_ids = []
    target_ids = []
    for i in tqdm(range(0, len(data), batch_size), ncols=100):
        input_buffer = []
        target_buffer = []
        for j in range(i, min(i+batch_size, len(data)), 1):
            input_, target = (
                data.loc[j, "sentence1"],
                data.loc[j, "sentence2"]
            )
            # The "paraphrase: " tokens are added later in the collate_fn
            # input_ = input_ + " </s>"
            # target = target + " </s>"
            input_buffer.append(input_)
            target_buffer.append(target)
        # tokenize inputs
        tokenized_inputs = tokenizer.batch_encode_plus(
            input_buffer
        )
        # tokenize targets
        tokenized_targets = tokenizer.batch_encode_plus(
            target_buffer
        )
        input_ids.extend(tokenized_inputs["input_ids"])
        target_ids.extend(tokenized_targets["input_ids"])
    return input_ids, target_ids


def main(dataset: Dataset, tokenizer_name: str = "t5-base", batch_size: int = 1024):
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    for datafile in (f"{dataset.value}_train.csv", f"{dataset.value}_valid.csv", f"{dataset.value}_test.csv"):
        print(datafile)
        data = pd.read_csv(DATA_PATH / datafile)
        input_ids, target_ids = process_file(data, tokenizer, batch_size)
        joblib.dump([input_ids, target_ids], CACHE_PATH / (Path(datafile).stem + ".jbl"))
        print(CACHE_PATH / (Path(datafile).stem + ".jbl"))


if __name__ == "__main__":
    typer.run(main)

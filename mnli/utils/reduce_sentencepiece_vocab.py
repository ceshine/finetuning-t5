import json
import shutil
from pathlib import Path
from itertools import chain
from typing import Set, Sequence

import typer
import pandas as pd
from tqdm import tqdm
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from sentencepiece import sentencepiece_model_pb2 as model


FILE_BLACKLIST = ("train.csv", "sample_submission.csv")


def tokenize_data(tokenizer: MT5Tokenizer, data: Sequence, seen: Set, batch_size=1024):
    for i in tqdm(range(0, len(data), batch_size)):
        seen = seen.union(
            set(chain.from_iterable(tokenizer.batch_encode_plus(
                data[i:(i+batch_size)], return_attention_mask=False)["input_ids"]))
        )
    return seen


def collect_tokens(tokenizer: MT5Tokenizer, kaggle: bool, mnli: bool):
    folders = []
    seen: Set[int] = set()
    if kaggle:
        folders.append(Path("data/kaggle"))
    if mnli:
        folders.append(Path("data/multinli"))
    for folder in folders:
        for filepath in folder.iterdir():
            if filepath.suffixes[-1] == ".csv" and not (filepath.name in FILE_BLACKLIST):
                print(filepath)
                df = pd.read_csv(filepath)
                seen = tokenize_data(tokenizer, df["premise"].values, seen)
                seen = tokenize_data(tokenizer, df["hypothesis"].values, seen)
                print(len(seen))
    return seen


def main(t5_model: str, kaggle: bool = True, mnli: bool = True):
    model_name = t5_model.split("/")[-1]
    Path("cache/").mkdir(exist_ok=True)
    target_path = f"cache/{model_name}/"
    if Path(target_path).exists():
        # Remove existing model
        shutil.rmtree(target_path)
    tokenizer = MT5Tokenizer.from_pretrained(t5_model)
    tokenizer.save_pretrained(target_path)
    tmp = MT5ForConditionalGeneration.from_pretrained(t5_model)
    tmp.save_pretrained(target_path)
    del tmp

    seen_tokens = collect_tokens(tokenizer, kaggle, mnli)

    m = model.ModelProto()
    m.ParseFromString(open(f"{target_path}spiece.model", 'rb').read())

    kept_pieces, i = [], len(m.pieces) - 1
    while len(m.pieces):
        piece = m.pieces.pop()
        if i < 259 or i in seen_tokens:
            kept_pieces.append(piece)
        i -= 1
    kept_pieces = list(reversed(kept_pieces))
    print("# of kept pieces:", len(kept_pieces))
    m.pieces.extend(kept_pieces)

    # backup
    Path(f"{target_path}spiece.model").rename(f"{target_path}spiece.model.old")
    # write new
    with open(f"{target_path}spiece.model", 'wb') as f:
        f.write(m.SerializeToString())

    kept_ids = sorted(list(seen_tokens.union(set(range(259)))))
    with open(f"{target_path}kept_ids.json", 'w') as fout:
        json.dump(kept_ids, fout)


if __name__ == "__main__":
    typer.run(main)

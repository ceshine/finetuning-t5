import typer
from pathlib import Path
from functools import partial

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    MT5ForConditionalGeneration, MT5Tokenizer, MT5Config,
    T5ForConditionalGeneration, T5Tokenizer, T5Config
)
from torch.utils.data import DataLoader

from preprocess.tokenize_dataset import Corpus
from train import XNLIDataset
from t2t import collate_batch


def main(
        model_path: str, corpus: Corpus = "kaggle", split_name: str = "valid",
        max_len: int = 128, batch_size: int = 32):
    if "mt5" in Path(model_path).stem:
        tokenizer = MT5Tokenizer.from_pretrained(model_path)
        # print(tokenizer.encode("</s>"))
        model = MT5ForConditionalGeneration(
            MT5Config.from_pretrained(model_path)
        ).eval()
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        # print(tokenizer.encode("</s>"))
        model = T5ForConditionalGeneration(
            T5Config.from_pretrained(model_path)
        ).eval()
    model.lm_head = torch.nn.Linear(model.lm_head.in_features, 3, bias=False)
    model.load_state_dict(torch.load(Path(model_path) / "pytorch_model.bin"))
    model = model.cuda()
    # model.load_state_dict(torch.load(model_path))
    context_tokens_1 = tokenizer.encode("mnli hypothesis:")[:-1]
    context_tokens_2 = tokenizer.encode("premise:")[:-1]
    collate_fn = partial(
        collate_batch, pad=model.config.decoder_start_token_id,
        decode_start_token=model.config.pad_token_id,
        max_len=max_len, is_classifier=True
    )
    dataset = XNLIDataset(
        corpus, split_name + ".jbl",
        context_tokens_1, context_tokens_2)
    data_loader = DataLoader(
        dataset, num_workers=1, shuffle=False, drop_last=False,
        batch_size=batch_size, collate_fn=collate_fn)
    preds, labels = [], []
    for input_batch, label_batch in tqdm(data_loader, ncols=100):
        for key, val in input_batch.items():
            input_batch[key] = val.cuda()
        outputs = model(**input_batch)
        preds_local = torch.argmax(outputs["logits"][:, 0, :].cpu(), dim=-1)
        preds.append(preds_local.numpy())
        labels.append(np.asarray([x[0] for x in label_batch["ids"].cpu().numpy()]))
    full_labels = np.concatenate(labels)
    full_preds = np.concatenate(preds)
    # print("Label mapping:")
    # for key in np.unique(full_labels):
    #     print(f"{key}: {tokenizer.decode([key])}")
    print("Labels:")
    print(pd.Series(full_labels).value_counts())
    print("Predictions:")
    print(pd.Series(full_preds).value_counts())
    print("Acc: %.2f%%" % (np.mean(full_labels == full_preds) * 100))


if __name__ == "__main__":
    typer.run(main)

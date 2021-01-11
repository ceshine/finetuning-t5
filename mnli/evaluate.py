import typer
from functools import partial

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader

from train import MNLIDataset, Corpus
from t2t import collate_batch


def main(model_path: str, corpus: Corpus = "kaggle", split_name: str = "valid"):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    # print(tokenizer.encode("</s>"))
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    # model.load_state_dict(torch.load(model_path))
    context_tokens_1 = tokenizer.encode("mnli hypothesis:")[:-1]
    context_tokens_2 = tokenizer.encode("premise:")[:-1]
    collate_fn = partial(
        collate_batch, pad=model.config.decoder_start_token_id,
        decode_start_token=model.config.pad_token_id,
        max_len=64, is_classifier=False
    )
    dataset = MNLIDataset(
        corpus, split_name + ".jbl",
        context_tokens_1, context_tokens_2)
    data_loader = DataLoader(
        dataset, num_workers=1, shuffle=False, drop_last=False,
        batch_size=8, collate_fn=collate_fn)
    preds, labels = [], []
    for input_batch, label_batch in tqdm(data_loader, ncols=100):
        outputs = model(**input_batch)
        preds_local = torch.argmax(outputs["logits"][:, 0, :], dim=-1)
        preds.append(preds_local.numpy())
        labels.append(np.asarray([x[0] for x in label_batch["ids"].numpy()]))
    full_labels = np.concatenate(labels)
    full_preds = np.concatenate(preds)
    print(pd.Series(full_labels).value_counts())
    print(pd.Series(full_preds).value_counts())
    print("Acc: %.2f%%" % (np.mean(full_labels == full_preds) * 100))


if __name__ == "__main__":
    typer.run(main)

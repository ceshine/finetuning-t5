from pathlib import Path
from functools import partial

import typer
import torch
import numpy as np
import pandas as pd
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, MT5Config
from torch.utils.data import DataLoader

from preprocess.tokenize_dataset import process_file
from train import XNLIDataset, shrink_vocab
from t2t import collate_batch


def main(model_path: str, data_folder: str = "/kaggle/input/contradictory-my-dear-watson/"):
    df_test = pd.read_csv(Path(data_folder) / "test.csv")

    tokenizer = MT5Tokenizer.from_pretrained(model_path)

    model = MT5ForConditionalGeneration(
        MT5Config.from_pretrained(model_path)
    )
    model.lm_head = torch.nn.Linear(model.lm_head.in_features, 3, bias=False)
    shrink_vocab(model_path, model)
    model.load_state_dict(torch.load(str(Path(model_path) / "pytorch_model.bin")))
    model = model.cuda().eval()

    # label_tokens_dict = {
    #     tokens[0]: idx for idx, tokens in enumerate(tokenizer.batch_encode_plus(
    #         ["entailment", "neutral", "contradiction"]
    #     )["input_ids"])
    # }

    class InferenceDataset(XNLIDataset):
        def __init__(self, premise_ids, hypothesis_ids):
            self.labels = None
            self.premise_ids = premise_ids
            self.hypothesis_ids = hypothesis_ids
            self.context_tokens_1 = tokenizer.encode("mnli hypothesis:")[:-1]
            self.context_tokens_2 = tokenizer.encode("premise:")[:-1]
            self.tokenizer = None
            self.max_len = 64

    collate_fn = partial(
        collate_batch, pad=model.config.decoder_start_token_id,
        decode_start_token=model.config.pad_token_id,
        max_len=128, is_classifier=False
    )

    premise_ids, hypothesis_ids, _ = process_file(df_test, tokenizer, batch_size=32)
    dataset = InferenceDataset(premise_ids, hypothesis_ids)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=collate_fn)

    preds = []
    for input_batch, _ in data_loader:
        for key, val in input_batch.items():
            input_batch[key] = val.cuda()
        outputs = model(**input_batch)
        preds_local = [
            x for x in torch.argmax(outputs["logits"][:, 0, :], dim=-1).cpu().numpy()
        ]
        preds.append(preds_local)

    preds = np.concatenate(preds)
    # inverse_label_dict = {key: tokenizer.decode([key]) for key in np.unique(preds)}

    df_sub = pd.DataFrame({
        "id": df_test.id.values,
        "prediction": preds  # [inverse_label_dict[x] for x in preds]
    })
    df_sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    typer.run(main)

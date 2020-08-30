"""
Reference: https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
"""
import os
import math
from functools import partial
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import typer
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_helper_bot import (
    BaseBot, LearningRateSchedulerCallback,
    MovingAverageStatsTrackerCallback,
    CheckpointCallback,
    # EarlyStoppingCallback,
    MultiStageScheduler, LinearLR,
    TelegramCallback
    # WandbCallback,
)
from pytorch_helper_bot.bot import batch_to_device
from pytorch_helper_bot.optimizers import RAdam
from transformers import T5ForConditionalGeneration, T5Tokenizer

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

CACHE_DIR = Path("cache/")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class T2TBot(BaseBot):
    log_dir: Path = CACHE_DIR / "logs/"

    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"

    @staticmethod
    def extract_prediction(output):
        return output[0]

    def eval(self, loader):
        """Override to avoid OOM"""
        self.model.eval()
        preds, ys = [], []
        losses, weights = [], []
        self.logger.debug("Evaluating...")
        with torch.no_grad():
            for *input_tensors, y_local in tqdm(loader, disable=not self.pbar, ncols=100):
                input_tensors, y_local = self.run_batch_inputs_callbacks(
                    input_tensors, y_local, is_eval=True)
                input_tensors = batch_to_device(input_tensors, self.device)
                y_local = batch_to_device([y_local], self.device)[0]
                if len(input_tensors) == 1 and isinstance(input_tensors[0], dict):
                    output = self.extract_prediction(
                        self.model(**input_tensors[0]))
                else:
                    output = self.extract_prediction(
                        self.model(*input_tensors))
                batch_loss = self.criterion(output, y_local)
                losses.append(batch_loss.data.cpu().item())
                weights.append(output.size(self.batch_dim))
                # Save batch labels and predictions
                # THE FOLLOWING TWO LINES WERE CHANGED
                preds.append(torch.argmax(output, dim=-1)[:, :y_local["ids"].size(1)].cpu())
                ys.append(batch_to_device([y_local], "cpu")[0])
        loss = np.average(losses, weights=weights)
        metrics = {"loss": (loss, self.loss_format % loss)}
        # THE FOLLOWING LINE WAS CHANGED BECAUSE THE SHAPES OF ys ARE NOT CONSTANT
        global_ys, global_preds = ys, preds
        for metric in self.metrics:
            metric_loss, metric_string = metric(global_ys, global_preds)
            metrics[metric.name] = (metric_loss, metric_string)
        return metrics


def masked_cross_entropy_loss(outputs, targets):
    targets, mask = targets["ids"], targets["mask"]
    # print(outputs.shape, targets.shape)
    loss = torch.sum(
        mask.view(-1) * F.cross_entropy(
            outputs.view(-1, outputs.size(2)),
            targets.view(-1),
            reduction="none"
        )
    ) / mask.sum()
    return loss


class ParaphraseDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_len=256, batch_size=1024):
        self.path = Path(filepath)
        self.batch_size = batch_size

        self.source_column = "question1"
        self.target_column = "question2"
        self.data = pd.read_csv(self.path)

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.input_ids = []
        self.input_masks = []
        self.target_ids = []
        self.target_masks = []

        self._build()

    def __len__(self):
        return len(self.input_ids) * 2

    def __getitem__(self, index):
        if index >= len(self.input_ids):
            # flip the input and target
            index = index - len(self.input_ids)
            return (
                self.target_ids[index], self.target_masks[index],
                self.input_ids[index], self.input_masks[index]
            )
        else:
            return (
                self.input_ids[index], self.input_masks[index],
                self.target_ids[index], self.target_masks[index]
            )
        # return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        # return (
        #     source_ids, src_mask, target_ids, target_mask,
        #     {"ids": target_ids, "mask": target_mask}
        # )

    def _build(self):
        for i in tqdm(range(0, len(self.data), self.batch_size), ncols=100):
            input_buffer = []
            target_buffer = []
            for j in range(i, min(i+self.batch_size, len(self.data)), 1):
                input_, target = self.data.loc[
                    j, self.source_column
                ], self.data.loc[j, self.target_column]
                input_ = "paraphrase: " + input_ + ' </s>'
                target = target + " </s>"
                input_buffer.append(input_)
                target_buffer.append(target)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                input_buffer, max_length=self.max_len,
                truncation=True, pad_to_max_length=True,
                # pad_to_multiple_of=8,
                return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                target_buffer, max_length=self.max_len,
                truncation=True, pad_to_max_length=True,
                # pad_to_multiple_of=8,
                return_tensors="pt"
            )
            self.input_ids.extend(tokenized_inputs["input_ids"])
            self.input_masks.extend(tokenized_inputs["attention_mask"])
            self.target_ids.extend(tokenized_targets["input_ids"])
            self.target_masks.extend(tokenized_targets["attention_mask"])


def optimize_sequence(ids, attention_mask):
    # Pad to the minimum multiple of 8 to utilize tensor cores
    max_length = math.ceil(
        attention_mask.sum(dim=1).max().numpy() / 8.
    ) * 8
    return ids[:, :max_length], attention_mask[:, :max_length]


def collate_batch(batch, pad=0):
    """Batch preparation.

    Truncate the sequence to reduce wastes.
    """
    source_ids, src_mask, target_ids, target_mask = map(
        lambda x: torch.stack(x), zip(*batch)
    )
    source_ids, src_mask = optimize_sequence(source_ids, src_mask)
    target_ids, target_mask = optimize_sequence(target_ids, target_mask)
    shifted_target_ids = target_ids.new_zeros(target_ids.shape)
    shifted_target_ids[..., 1:] = target_ids[..., :-1].clone()
    shifted_target_ids[..., 0] = pad
    # print(source_ids.shape, src_mask.shape, target_ids.shape, target_mask.shape)
    return (
        {
            "input_ids": source_ids, "attention_mask": src_mask,
            "decoder_input_ids": shifted_target_ids
        },
        {"ids": target_ids, "mask": target_mask}
    )


def main(t5_model: str = "t5-base", lr: float = 1e-4, steps: Optional[int] = None, amp_level: str = ""):
    model = T5ForConditionalGeneration.from_pretrained(t5_model).cuda()
    tokenizer = T5Tokenizer.from_pretrained(t5_model)
    # print(model.config.decoder_start_token_id)
    # print(tokenizer.pad_token_id)
    train_dataset = ParaphraseDataset(tokenizer, 'data/quora_train.csv', 256)
    valid_dataset = ParaphraseDataset(tokenizer, 'data/quora_valid.csv', 256)
    print("Train dataset: ", len(train_dataset))
    print("Valid dataset: ", len(valid_dataset))
    train_loader = DataLoader(
        train_dataset, num_workers=0,
        batch_size=16, collate_fn=partial(collate_batch, pad=model.config.decoder_start_token_id))
    valid_loader = DataLoader(
        valid_dataset, num_workers=0,
        batch_size=24, collate_fn=partial(collate_batch, pad=model.config.decoder_start_token_id))
    optimizer = RAdam(
        model.parameters(), lr=lr
    )
    if amp_level:
        if not APEX_AVAILABLE:
            raise ValueError("Apex is not installed!")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=amp_level
        )
    if steps is None:
        # By default train for 2 epochs
        steps = len(train_loader) * 2
    lr_durations = [
        int(steps*0.1),
        int(np.ceil(steps*0.9))
    ]
    break_points = [0] + list(np.cumsum(lr_durations))[:-1]
    checkpoints = CheckpointCallback(
        keep_n_checkpoints=1,
        checkpoint_dir=CACHE_DIR / "model_cache/",
        monitor_metric="loss"
    )
    callbacks = [
        MovingAverageStatsTrackerCallback(
            avg_window=len(train_loader) // 8,
            log_interval=len(train_loader) // 10
        ),
        LearningRateSchedulerCallback(
            MultiStageScheduler(
                [
                    LinearLR(optimizer, 0.01, lr_durations[0]),
                    CosineAnnealingLR(optimizer, lr_durations[1], eta_min=1e-8)
                ],
                start_at_epochs=break_points
            )
        ),
        checkpoints,
        TelegramCallback(
            token=os.environ.get("TELEGRAM_TOKEN"),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID"), name="T5",
            report_evals=True
        ),
        # EarlyStoppingCallback(
        #     patience=8, min_improv=1e-2,
        #     monitor_metric="accuracy"
        # ),
        # WandbCallback(
        #     config={
        #         "epochs": args.epochs,
        #         "arch": args.arch
        #     },
        #     name="Imagenatte",
        #     watch_freq=200,
        #     watch_level="gradients"
        # )
    ]
    # for batch in valid_loader:
    #     print(batch[0].size(), batch[1].size())
    bot = T2TBot(
        model=model, train_loader=train_loader,
        valid_loader=valid_loader, clip_grad=10.,
        optimizer=optimizer, echo=True,
        criterion=masked_cross_entropy_loss,
        callbacks=callbacks,
        pbar=True, use_tensorboard=True,
        use_amp=(amp_level != '')
    )
    bot.train(
        total_steps=steps,
        checkpoint_interval=len(train_loader) // 5
    )
    bot.load_model(checkpoints.best_performers[0][1])
    torch.save(bot.model.state_dict(), CACHE_DIR /
               "final_weights.pth")
    checkpoints.remove_checkpoints(keep=0)


if __name__ == "__main__":
    typer.run(main)

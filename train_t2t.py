"""
Reference: https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
"""
import enum
import math
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import typer
from pytorch_helper_bot.bot import batch_to_device
from pytorch_helper_bot.optimizers import RAdam
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

from pytorch_helper_bot import (  # EarlyStoppingCallback,; WandbCallback,
    BaseBot, CheckpointCallback, LearningRateSchedulerCallback, LinearLR,
    MovingAverageStatsTrackerCallback, MultiStageScheduler, TelegramCallback)

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

CACHE_DIR = Path("cache/")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


class Corpus(enum.Enum):
    QUORA = "quora"
    PAWS = "paws"
    MSRP = "msrp"
    OPINOSIS = "opinosis"
    QP = "quora+paws"
    PMO = "paws+msrp+opinosis"


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
    def __init__(self, corpus, file_suffix: str, context_tokens: List[int]):
        input_ids, target_ids = [], []
        if corpus in (Corpus.PAWS, Corpus.QP, Corpus.PMO):
            tmp = joblib.load(CACHE_DIR / f'paws{file_suffix}')
            multiplier = 1
            if corpus is Corpus.QP:
                # oversample PAWS
                multiplier = 2
            input_ids.extend(tmp[0] * multiplier)
            target_ids.extend(tmp[1] * multiplier)
        if corpus in (Corpus.QUORA, Corpus.QP):
            tmp = joblib.load(CACHE_DIR / f'quora{file_suffix}')
            input_ids.extend(tmp[0])
            target_ids.extend(tmp[1])
        if corpus in (Corpus.MSRP, Corpus.PMO):
            tmp = joblib.load(CACHE_DIR / f'msrp{file_suffix}')
            multiplier = 1
            if corpus is Corpus.PMO:
                # oversample MSRP
                multiplier = 2
            input_ids.extend(tmp[0] * multiplier)
            target_ids.extend(tmp[1] * multiplier)
        if corpus in (Corpus.OPINOSIS, Corpus.PMO):
            tmp = joblib.load(CACHE_DIR / f'opinosis{file_suffix}')
            multiplier = 1
            if corpus is Corpus.PMO:
                # oversample MSRP
                multiplier = 2
            input_ids.extend(tmp[0] * multiplier)
            target_ids.extend(tmp[1] * multiplier)
        self.input_ids, self.target_ids = input_ids, target_ids
        self.context_tokens = context_tokens

    def __len__(self):
        return len(self.input_ids) * 2

    def __getitem__(self, index):
        if index >= len(self.input_ids):
            # flip the input and target
            index = index - len(self.input_ids)
            return (
                torch.tensor(self.context_tokens + self.target_ids[index], dtype=torch.int64),
                torch.tensor(self.input_ids[index], dtype=torch.int64)
            )
        else:
            return (
                torch.tensor(self.context_tokens + self.input_ids[index], dtype=torch.int64),
                torch.tensor(self.target_ids[index], dtype=torch.int64)
            )


def optimize_sequence(ids, pad, max_len):
    # Pad to the minimum multiple of 8 to utilize tensor cores
    max_length = math.ceil(
        min(max_len, np.max([x.size(0) for x in ids])) / 8.
    ) * 8
    padded_ids = ids[0].new_zeros((len(ids), max_length)) + pad
    mask = ids[0].new_zeros((len(ids), max_length))
    for i, example in enumerate(ids):
        example = example[:max_len]
        padded_ids[i, :len(example)] = example
        mask[i, :len(example)] = 1
    return padded_ids, mask


def collate_batch(batch, max_len, pad=0, decode_start_token=0):
    """Batch preparation.

    Truncate the sequence to reduce wastes.
    """
    source_ids, target_ids = zip(*batch)
    source_ids, src_mask = optimize_sequence(source_ids, pad, max_len)
    target_ids, target_mask = optimize_sequence(target_ids, pad, max_len)
    shifted_target_ids = target_ids.new_zeros(target_ids.shape)
    shifted_target_ids[..., 1:] = target_ids[..., :-1].clone()
    shifted_target_ids[..., 0] = decode_start_token
    # print(source_ids.shape, src_mask.shape, target_ids.shape, target_mask.shape)
    return (
        {
            "input_ids": source_ids, "attention_mask": src_mask,
            "decoder_input_ids": shifted_target_ids
        },
        {"ids": target_ids, "mask": target_mask}
    )


def test_run(model, optimizer, batch_size, max_len, loss_fn, use_amp: bool):
    dummy_tensor = torch.ones((batch_size, max_len)).long().cuda()
    print(dummy_tensor.shape)
    for _ in range(5):
        outputs = model(
            input_ids=dummy_tensor.clone(),
            attention_mask=dummy_tensor.clone(),
            decoder_input_ids=dummy_tensor.clone()
        )
        loss = loss_fn(outputs[0], {
            "ids": dummy_tensor.clone(),
            "mask": dummy_tensor.clone()
        })
        if use_amp:
            with amp.scale_loss(
                loss, optimizer
            ) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    # do not actually use the gradients
    optimizer.zero_grad()


def main(
    t5_model: str = "t5-base", lr: float = 1e-4,
    steps: Optional[int] = None, amp_level: str = "",
    dataset: Corpus = Corpus.PAWS, batch_size: int = 16,
    max_len: int = 64, grad_accu: int = 1
):
    model = T5ForConditionalGeneration.from_pretrained(t5_model).cuda()
    tokenizer = T5Tokenizer.from_pretrained(t5_model)
    context_tokens = tokenizer.encode("paraphrase: ")
    # print(model.config.decoder_start_token_id)
    # print(tokenizer.pad_token_id)
    train_dataset = ParaphraseDataset(dataset, '_train.jbl', context_tokens)
    valid_dataset = ParaphraseDataset(dataset, '_valid.jbl', context_tokens)
    print("Train dataset: ", len(train_dataset))
    print("Valid dataset: ", len(valid_dataset))
    collate_fn = partial(
        collate_batch, pad=model.config.decoder_start_token_id,
        decode_start_token=model.config.pad_token_id,
        max_len=max_len
    )
    train_loader = DataLoader(
        train_dataset, num_workers=0, shuffle=True, drop_last=True,
        batch_size=batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(
        valid_dataset, num_workers=0, shuffle=False, drop_last=False,
        batch_size=batch_size*2, collate_fn=collate_fn)
    optimizer = RAdam(
        model.parameters(), lr=lr
    )
    if amp_level:
        if not APEX_AVAILABLE:
            raise ValueError("Apex is not installed!")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=amp_level
        )
    test_run(model, optimizer, batch_size, max_len=max_len, loss_fn=masked_cross_entropy_loss, use_amp=amp_level != "")
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
            avg_window=steps // 16,
            log_interval=steps // 20
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
        checkpoints
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
    if os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        callbacks.append(TelegramCallback(
            token=os.environ.get("TELEGRAM_TOKEN"),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID"), name="T5",
            report_evals=True
        ))
    # for batch in valid_loader:
    #     print(batch[0].size(), batch[1].size())
    bot = T2TBot(
        model=model, train_loader=train_loader,
        valid_loader=valid_loader, clip_grad=10.,
        optimizer=optimizer, echo=True,
        criterion=masked_cross_entropy_loss,
        callbacks=callbacks,
        pbar=True, use_tensorboard=True,
        use_amp=(amp_level != ''),
        gradient_accumulation_steps=grad_accu
    )
    bot.train(
        total_steps=steps,
        checkpoint_interval=steps // 5
    )
    bot.load_model(checkpoints.best_performers[0][1])
    # torch.save(bot.model.state_dict(), CACHE_DIR /
    #            "final_weights.pth")
    target_dir = CACHE_DIR / "t5-finetuned"
    target_dir.mkdir(exist_ok=True, parents=True)
    bot.model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    checkpoints.remove_checkpoints(keep=0)


if __name__ == "__main__":
    typer.run(main)

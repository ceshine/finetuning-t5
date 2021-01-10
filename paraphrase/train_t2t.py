"""
Reference: https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
"""
import enum
import math
import os
from pathlib import Path
from functools import partial
from typing import List, Callable
from dataclasses import dataclass, asdict

import typer
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
from transformers import T5ForConditionalGeneration, T5Tokenizer

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
class Config:
    base_t5_model: str
    batch_size: int
    fp16: bool
    dataset: Corpus
    learning_rate: float
    weight_decay: float
    epochs: int
    max_len: int
    loss_fn: Callable
    num_gpus: int = 1
    grad_accu: int = 1


class T5Model(pl.LightningModule):
    def __init__(self, config: Config, **kwargs):
        super().__init__()
        self.config = config
        # log the config values
        self.save_hyperparameters(asdict(config))
        self.model = T5ForConditionalGeneration.from_pretrained(config.base_t5_model)
        self.tokenizer = T5Tokenizer.from_pretrained(config.base_t5_model)
        self.context_tokens = self.tokenizer.encode("paraphrase: ")
        self.collate_fn = partial(
            collate_batch, pad=self.model.config.decoder_start_token_id,
            decode_start_token=self.model.config.pad_token_id,
            max_len=self.config.max_len
        )
        self.metrics = [
            ("acc", pl.metrics.Accuracy(compute_on_step=False))
        ]
        self.train_loss_tracker = pls.utils.EMATracker(alpha=0.02)
        self.train_dataset = ParaphraseDataset(self.config.dataset, '_train.jbl', self.context_tokens)
        print("Train dataset: ", len(self.train_dataset))
        self.valid_dataset = ParaphraseDataset(self.config.dataset, '_valid.jbl', self.context_tokens)
        print("Valid dataset: ", len(self.valid_dataset))

    def forward(self, input_tensors):
        return self.model(**input_tensors)[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, num_workers=1, shuffle=True, drop_last=True,
            batch_size=self.config.batch_size, collate_fn=self.collate_fn)

    def get_progress_bar_dict(self):
        # don't show the experiment version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, num_workers=1, shuffle=False, drop_last=False,
            batch_size=self.config.batch_size*2, collate_fn=self.collate_fn)

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch[0])
        loss = self.config.loss_fn(
            logits,
            batch[1]
        )
        preds = torch.argmax(logits, dim=-1)[:, :batch[1]["ids"].size(1)]
        return {
            'loss': loss,
            'preds': preds,
            'target': batch[1]
        }

    def validation_step_end(self, outputs):
        self.log('val_loss', outputs['loss'].mean())
        for name, metric in self.metrics:
            metric(
                outputs['preds'].view(-1).cpu(),
                outputs['target']['ids'].view(-1).cpu()
            )
            self.log("val_" + name, metric)

    def _should_log(self, flag):
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if isinstance(flag, list):
                return flag[0]
            return flag
        return False

    def training_step_end(self, outputs):
        loss = outputs["loss"].mean()
        self.train_loss_tracker.update(loss.detach())
        if self._should_log(outputs["log"]):
            self.logger.log_metrics({
                "train_loss": self.train_loss_tracker.value
            }, step=self.global_step)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.config.loss_fn(
            self.forward(batch[0]),
            batch[1]
        )
        return {"loss": loss, "log": batch_idx % self.trainer.accumulate_grad_batches == 0}

    def configure_optimizers(self):
        optimizer = pls.optimizers.RAdam(
            self.model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        steps_per_epochs = math.floor(
            len(self.train_dataset) / self.config.batch_size / self.config.grad_accu  # / self.num_gpus # dpp mode
        )
        print("Steps per epochs:", steps_per_epochs)
        n_steps = steps_per_epochs * self.config.epochs
        lr_durations = [
            int(n_steps*0.05),
            int(np.ceil(n_steps*0.95)) + 1
        ]
        break_points = [0] + list(np.cumsum(lr_durations))[:-1]
        scheduler = {
            'scheduler': pls.lr_schedulers.MultiStageScheduler(
                [
                    pls.lr_schedulers.LinearLR(optimizer, 0.01, lr_durations[0]),
                    CosineAnnealingLR(optimizer, lr_durations[1])
                ],
                start_at_epochs=break_points
            ),
            'interval': 'step',
            'frequency': 1,
            'strict': True,
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


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
    def __init__(self, corpus: Corpus, file_suffix: str, context_tokens: List[int]):
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


def main(
    t5_model: str = "t5-base", lr: float = 1e-4,
    epochs: int = 5, fp16: bool = False,
    dataset: Corpus = Corpus.PAWS, batch_size: int = 16,
    max_len: int = 64, grad_accu: int = 1,
    num_gpus: int = 1
):
    pl.seed_everything(int(os.environ.get("SEED", 738)))
    config = Config(
        base_t5_model=t5_model,
        learning_rate=lr,
        epochs=epochs,
        dataset=dataset,
        max_len=max_len,
        grad_accu=grad_accu,
        batch_size=batch_size,
        fp16=fp16,
        weight_decay=0,
        num_gpus=num_gpus,
        loss_fn=masked_cross_entropy_loss
    )

    pl_module = T5Model(config)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(CACHE_DIR / "model_checkpoints"),
            monitor='val_loss',
            mode="min",
            filename='{step:06d}-{val_loss:.4f}',
            save_top_k=1,
            save_last=False
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]
    trainer = pl.Trainer(
        accelerator='dp' if num_gpus > 1 else None,
        # amp_backend="apex", amp_level='O2',
        precision=16 if config.fp16 else 32,
        gpus=config.num_gpus,
        val_check_interval=0.25,
        gradient_clip_val=10,
        max_epochs=epochs,
        # max_steps=steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_accu,
        # auto_scale_batch_size='power' if batch_size is None else None,
        logger=[
            pl.loggers.TensorBoardLogger(str(CACHE_DIR / "tb_logs"), name=""),
            pls.loggers.ScreenLogger(),
            # pl.loggers.WandbLogger(project="t5-paraphrase")
        ],
        log_every_n_steps=100
    )

    trainer.fit(pl_module)

    # pl_module.model.save_pretrained(CACHE_DIR / f"{config.base_t5_model}_last")
    # pl_module.tokenizer.save_pretrained(CACHE_DIR / f"{config.base_t5_model}_last")
    # print("Last model saved")

    assert isinstance(callbacks[0], pl.callbacks.ModelCheckpoint)
    print(callbacks[0].best_model_path)
    pl_module = T5Model.load_from_checkpoint(
        callbacks[0].best_model_path,
        config=config
    )
    pl_module.model.save_pretrained(CACHE_DIR / f"{config.base_t5_model}_best")
    pl_module.tokenizer.save_pretrained(CACHE_DIR / f"{config.base_t5_model}_best")
    print("Best model saved")


if __name__ == "__main__":
    typer.run(main)

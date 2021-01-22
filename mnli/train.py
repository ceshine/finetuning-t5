import os
import gc
import math
from itertools import chain
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

import typer
import torch
import joblib
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    MT5ForConditionalGeneration, MT5Tokenizer, T5ForConditionalGeneration, T5Tokenizer,
    MT5Config, T5Config, Adafactor
)

from t2t import BaseConfig, T5BaseModel, single_token_cross_entropy_loss
from preprocess.tokenize_dataset import Corpus

CACHE_DIR = Path("cache/")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class Config(BaseConfig):
    dataset: Corpus = Corpus.KAGGLE
    decoder_only: bool = True
    num_classes: int = 3


def load_model(model_class, model_config_class, config):
    model_path = config.base_t5_model
    try:
        model = model_class.from_pretrained(model_path)
        # replace the lm_head
        model.lm_head = torch.nn.Linear(model.lm_head.in_features, config.num_classes, bias=False)
    except RuntimeError:
        model = model_class(
            model_config_class.from_pretrained(model_path)
        )
        model.lm_head = torch.nn.Linear(model.lm_head.in_features, config.num_classes, bias=False)
        model.load_state_dict(torch.load(Path(model_path) / "pytorch_model.bin"))
    return model


class T5Model(T5BaseModel):
    def __init__(self, config: Config, **kwargs):
        if "mt5" in config.base_t5_model:
            tokenizer = MT5Tokenizer.from_pretrained(config.base_t5_model)
            model = load_model(MT5ForConditionalGeneration, MT5Config, config)
            # tie the weights
            # model.lm_head.weight = model.shared.weight
        else:
            tokenizer = T5Tokenizer.from_pretrained(config.base_t5_model)
            model = load_model(T5ForConditionalGeneration, T5Config, config)
        super().__init__(config, model, tokenizer, is_classifier=True)
        self.config = config
        # log the config values
        self.save_hyperparameters(asdict(config))
        self.context_tokens_1 = self.tokenizer.encode("xnli hypothesis:")[:-1]
        self.context_tokens_2 = self.tokenizer.encode("premise:")[:-1]
        self.train_dataset = XNLIDataset(
            self.config.dataset, 'train_split.jbl',
            self.context_tokens_1, self.context_tokens_2,
            max_len=config.max_len // 2
        )  # , tokenizer)
        print("Train dataset: ", len(self.train_dataset))
        self.valid_dataset = XNLIDataset(
            self.config.dataset, 'valid.jbl', self.context_tokens_1, self.context_tokens_2,
            max_len=config.max_len // 2
        )
        print("Valid dataset: ", len(self.valid_dataset))

    def configure_optimizers(self):
        if self.config.decoder_only:
            pls.utils.set_trainable(self.model.encoder.block, False)
            pls.utils.set_trainable(self.model.encoder.final_layer_norm, False)
            params = self.model.lm_head.parameters()
            if not (self.model.lm_head.weight is self.model.shared.weight):
                # weight is not shared
                pls.utils.set_trainable(self.model.shared, False)
                params = chain(
                    self.model.decoder.block.parameters(),
                    self.model.decoder.final_layer_norm.parameters(),
                    self.model.lm_head.parameters()
                )
            else:
                pls.utils.set_trainable(self.model.decoder.block, False)
                pls.utils.set_trainable(self.model.decoder.final_layer_norm, False)
                params = self.model.lm_head.parameters()
            optimizer = Adafactor(  # torch.optim.AdamW(
                [
                    {
                        "params": params,
                        # "learning_rate": self.config.learning_rate,
                        # "weight_decay": self.config.weight_decay

                    }
                ],
                relative_step=True,
                warmup_init=True, clip_threshold=1.0,  # lr=self.config.learning_rate,
                scale_parameter=True
            )
        else:
            # # make sure the weights are tied
            # assert self.model.lm_head.weight is self.model.shared.weight, (
            #     self.model.shared.weight - self.model.lm_head.weight).sum()
            optimizer = Adafactor(
                self.model.parameters(),
                relative_step=True,
                warmup_init=True, clip_threshold=1.0,  # lr=self.config.learning_rate,
                scale_parameter=True
            )
            #     [
            #         {
            #             "params": (
            #                 chain(
            #                     self.model.encoder.block.parameters(),
            #                     self.model.encoder.final_layer_norm.parameters()
            #                 ) if (self.model.lm_head.weight is self.model.shared.weight) else
            #                 self.model.encoder.parameters()
            #             ),
            #             "learning_rate": self.config.learning_rate / 2,
            #             "weight_decay": self.config.weight_decay / 2,
            #         },
            #         {
            #             "params": chain(
            #                 self.model.decoder.block.parameters(),
            #                 self.model.decoder.final_layer_norm.parameters(),
            #                 self.model.lm_head.parameters()
            #             ),
            #             "learning_rate": self.config.learning_rate,
            #             "weight_decay": self.config.weight_decay

            #         }
            #     ]
            # )
        print("Optimizer parameter count: {:,d}".format(np.sum([
            pls.utils.count_parameters(group["params"]) for group in optimizer.param_groups
        ])))
        # steps_per_epochs = math.floor(
        #     len(self.train_dataset) / self.config.batch_size / self.config.grad_accu  # / self.num_gpus # dpp mode
        # )
        # print("Steps per epochs:", steps_per_epochs)
        # n_steps = steps_per_epochs * self.config.epochs
        # lr_durations = [
        #     int(n_steps*0.1),
        #     int(np.ceil(n_steps*0.9)) + 1
        # ]
        # break_points = [0] + list(np.cumsum(lr_durations))[:-1]
        # scheduler = {
        #     'scheduler': pls.lr_schedulers.MultiStageScheduler(
        #         [
        #             pls.lr_schedulers.LinearLR(optimizer, 0.0001, lr_durations[0]),
        #             CosineAnnealingLR(optimizer, lr_durations[1])
        #         ],
        #         start_at_epochs=break_points
        #     ),
        #     'interval': 'step',
        #     'frequency': 1,
        #     'strict': True,
        # }
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler
        }


class XNLIDataset(Dataset):
    def __init__(
            self, corpus: Corpus, file_name: str, context_tokens_1: List[int], context_tokens_2: List[int],
            max_len: int = 128, tokenizer=None):
        self.premise_ids, self.hypothesis_ids, self.labels = joblib.load(
            CACHE_DIR / corpus.value / f'{file_name}')
        self.context_tokens_1 = context_tokens_1
        self.context_tokens_2 = context_tokens_2
        self.max_len = max_len  # max length for premise and hypothesis
        # for debug
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.premise_ids)

    def __getitem__(self, index):
        if self.labels is None:
            label = torch.tensor([0], dtype=torch.int64)
        else:
            label = torch.tensor(self.labels[index], dtype=torch.int64)
        premise_length = min(self.max_len, len(self.premise_ids[index]))
        hypothesis_length = min(self.max_len-1, len(self.hypothesis_ids[index]) - 1)
        if self.tokenizer:
            print(self.tokenizer.decode(self.context_tokens_1 + self.hypothesis_ids[index][:hypothesis_length] +
                                        self.context_tokens_2 + self.premise_ids[index][:premise_length]))
            print(self.tokenizer.decode(self.labels[index]))
        return (
            torch.tensor(
                self.context_tokens_1 + self.hypothesis_ids[index][:-1][:hypothesis_length] +
                self.context_tokens_2 + self.premise_ids[index][:premise_length],
                dtype=torch.int64
            ),
            label
        )


def main(
    t5_model: str = "google/mt5-small", lr: float = 1e-4,
    epochs: int = 5, fp16: bool = False,
    dataset: Corpus = "kaggle", batch_size: int = 16,
    max_len: int = 64, grad_accu: int = 1,
    num_gpus: int = 1, disable_progress_bar: bool = False,
    valid_frequency: Optional[float] = None,
    full_model: bool = False, tpu_cores: int = 0


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
        num_gpus=num_gpus if tpu_cores == 0 else 0,
        tpu_cores=tpu_cores,
        loss_fn=single_token_cross_entropy_loss,
        decoder_only=not full_model
    )
    # print(config)

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
        # amp_backend="apex", amp_level='O1',
        precision=16 if config.fp16 else 32,
        gpus=config.num_gpus,
        val_check_interval=valid_frequency if valid_frequency else 1.0,
        # gradient_clip_val=3,
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
        log_every_n_steps=100,
        progress_bar_refresh_rate=0 if disable_progress_bar else 20,
        tpu_cores=config.tpu_cores if config.tpu_cores else None
    )

    trainer.fit(pl_module)

    model_name = config.base_t5_model.split("/")[-1]

    assert isinstance(callbacks[0], pl.callbacks.ModelCheckpoint)
    pl_module.load_state_dict(torch.load(callbacks[0].best_model_path)["state_dict"])
    del trainer
    gc.collect()
    pl_module.model.save_pretrained(CACHE_DIR / f"{model_name}_best")
    pl_module.tokenizer.save_pretrained(CACHE_DIR / f"{model_name}_best")
    print("Best model saved")


if __name__ == "__main__":
    typer.run(main)

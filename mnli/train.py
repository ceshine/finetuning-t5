import enum
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict

import typer
import joblib
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

from t2t import BaseConfig, T5BaseModel, masked_cross_entropy_loss

CACHE_DIR = Path("cache/")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


class Corpus(enum.Enum):
    KAGGLE = "kaggle"


@dataclass
class Config(BaseConfig):
    dataset: Corpus = Corpus.KAGGLE


class T5Model(T5BaseModel):
    def __init__(self, config: Config, **kwargs):
        model = MT5ForConditionalGeneration.from_pretrained(config.base_t5_model)
        tokenizer = MT5Tokenizer.from_pretrained(config.base_t5_model)
        super().__init__(config, model, tokenizer)
        self.config = config
        # log the config values
        self.save_hyperparameters(asdict(config))
        self.context_tokens_1 = self.tokenizer.encode("mnli hypothesis:")[:-1]
        self.context_tokens_2 = self.tokenizer.encode("premise:")[:-1]
        self.train_dataset = MNLIDataset(
            self.config.dataset, 'train_split.jbl',
            self.context_tokens_1, self.context_tokens_2)
        print("Train dataset: ", len(self.train_dataset))
        self.valid_dataset = MNLIDataset(
            self.config.dataset, 'valid.jbl', self.context_tokens_1, self.context_tokens_2)
        print("Valid dataset: ", len(self.valid_dataset))


class MNLIDataset(Dataset):
    def __init__(self, corpus: Corpus, file_name: str, context_tokens_1: List[int], context_tokens_2: List[int]):
        self.premise_ids, self.hypothesis_ids, self.labels = joblib.load(
            CACHE_DIR / corpus.value / f'{file_name}')
        self.context_tokens_1 = context_tokens_1
        self.context_tokens_2 = context_tokens_2

    def __len__(self):
        return len(self.premise_ids)

    def __getitem__(self, index):
        if self.labels is None:
            label = torch.tensor([0], dtype=torch.int64)
        else:
            label = torch.tensor(self.labels[index], dtype=torch.int64)
        return (
            torch.tensor(
                self.context_tokens_1 + self.hypothesis_ids[index][:-1] +
                self.context_tokens_2 + self.premise_ids[index],
                dtype=torch.int64
            ),
            label
        )


def main(
    t5_model: str = "google/mt5-small", lr: float = 1e-4,
    epochs: int = 5, fp16: bool = False,
    dataset: Corpus = "kaggle", batch_size: int = 16,
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

    model_name = config.base_t5_model.split("/")[-1]

    assert isinstance(callbacks[0], pl.callbacks.ModelCheckpoint)
    print(callbacks[0].best_model_path)
    pl_module = T5Model.load_from_checkpoint(
        callbacks[0].best_model_path,
        config=config
    )
    pl_module.model.save_pretrained(CACHE_DIR / f"{model_name}_best")
    pl_module.tokenizer.save_pretrained(CACHE_DIR / f"{model_name}_best")
    print("Best model saved")


if __name__ == "__main__":
    typer.run(main)

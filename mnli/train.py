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
    MT5Config, T5Config  # , Adafactor
)

from t2t import BaseConfig, T5BaseModel, single_token_cross_entropy_loss
from preprocess.tokenize_dataset import Corpus

CACHE_DIR = Path("cache/")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


class Adafactor(torch.optim.Optimizer):
    """
    AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost` https://arxiv.org/abs/1804.04235 Note that
    this optimizer internally adjusts the learning rate depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`):
            The external learning rate.
        eps (:obj:`Tuple[float, float]`, `optional`, defaults to (1e-30, 1e-3)):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (:obj:`float`, `optional`, defaults 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (:obj:`float`, `optional`, defaults to -0.8):
            Coefficient used to compute running averages of square
        beta1 (:obj:`float`, `optional`):
            Coefficient used for computing running averages of gradient
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Weight decay (L2 penalty)
        scale_parameter (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If True, learning rate is scaled by root mean square
        relative_step (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used

    This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

    Recommended T5 finetuning settings:

        - Scheduled LR warm-up to fixed LR
        - disable relative updates
        - use clip threshold: https://arxiv.org/abs/2004.14546

        Example::

            Adafactor(model.parameters(), lr=1e-3, relative_step=False, warmup_init=True)

        - Alternatively, relative_step with warmup_init can be used.
        - Training without LR warmup or clip threshold is not recommended. Additional optimizer operations like
          gradient clipping should not be used alongside Adafactor.

    Usage::

        # replace AdamW with Adafactor
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_()
        c_factor = exp_avg_sq_col.rsqrt()
        return torch.mm(r_factor.unsqueeze(-1), c_factor.unsqueeze(0))

    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            flag = False
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)
                if state["step"] % 100 == 0 and flag is False:
                    print(group["lr"])
                    flag = True

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-1))
                    exp_avg_sq_col.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-2))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(1.0 - beta2t, update)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(group["lr"])

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(1 - group["beta1"], update)
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss


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
                relative_step=True, warmup_init=True,
                clip_threshold=1.0, lr=self.config.learning_rate,
                scale_parameter=True
            )
        else:
            # # make sure the weights are tied
            # assert self.model.lm_head.weight is self.model.shared.weight, (
            #     self.model.shared.weight - self.model.lm_head.weight).sum()
            optimizer = Adafactor(
                self.model.parameters(),
                relative_step=True, warmup_init=True,
                clip_threshold=1.0, lr=self.config.learning_rate,
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
        steps_per_epochs = math.floor(
            len(self.train_dataset) / self.config.batch_size / self.config.grad_accu  # / self.num_gpus # dpp mode
        )
        print("Steps per epochs:", steps_per_epochs)
        # n_steps = steps_per_epochs * self.config.epochs
        # lr_durations = [
        #     int(n_steps*0.2),
        #     int(np.ceil(n_steps*0.8)) + 1
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

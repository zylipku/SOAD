import os
import sys

import lightning as L
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.nn import functional as F
from torchmetrics import MeanMetric, MetricCollection

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from diffusion.scheduler import VPSDEScheduler

from ..data.qg_datamodule import QGDataModule
from .components.unet.unet_sda_seq2d import UNetSDA


class SdaLitModule(L.LightningModule):
    def __init__(
        self,
        num_channels: int,
        noise_estimator: UNetSDA,
        diffusion_scheduler: VPSDEScheduler,
        optimizer: torch.optim.Optimizer,
        scheduler_kwargs: dict,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "noise_estimator",
                "diffusion_scheduler",
            ],
        )

        self.noise_estimator = noise_estimator
        self.diffusion_scheduler = diffusion_scheduler

        metrics = MetricCollection(
            {
                "loss_l1": MeanMetric(),
                "loss_l2": MeanMetric(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="valid/")

    def training_step(self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x = batch[0]
        # x.shape = (b, L, c, h, w)

        # ? diffusion training
        t = self.diffusion_scheduler.sample_timesteps(x.shape[0]).to(x.device)
        xt, noise = self.diffusion_scheduler.add_noise(x, t)
        noise_pred = self.noise_estimator(xt, t)

        loss_l2 = F.mse_loss(noise_pred, noise)
        loss_l1 = (noise_pred - noise).abs().mean()  # L1 noise

        self.train_metrics["loss_l1"](loss_l1)
        self.train_metrics["loss_l2"](loss_l2)

        self.log("train/loss_l1", self.train_metrics["loss_l1"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_l2", self.train_metrics["loss_l2"], on_step=True, on_epoch=False, sync_dist=True)

        return loss_l2

    def validation_step(self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x = batch[0]
        # x.shape = (b, L, c, h, w)

        # ? diffusion training
        t = self.diffusion_scheduler.sample_timesteps(x.shape[0]).to(x.device)
        xt, noise = self.diffusion_scheduler.add_noise(x, t)
        noise_pred = self.noise_estimator(xt, t)

        loss_l2 = F.mse_loss(noise_pred, noise)
        loss_l1 = (noise_pred - noise).abs().mean()  # L1 noise

        self.valid_metrics["loss_l1"](loss_l1)
        self.valid_metrics["loss_l2"](loss_l2)

        self.log("valid/loss_l1", self.valid_metrics["loss_l1"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/loss_l2", self.valid_metrics["loss_l2"], on_step=True, on_epoch=False, sync_dist=True)

        return loss_l2

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler_kwargs is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda t: 1 - (t / self.hparams.scheduler_kwargs["max_epochs"]),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    seed_everything(2357)

    n_epochs = 1024

    bs = 2

    hparams = {
        "dm": {
            "batch_size": bs,
            "seq_length": 9,
            "random_crop": True,
            "num_workers": 4,
        },
        "arch": {
            "channels": 2,
        },
        "opt": {
            # 'hparams': {
            #     'beta_warmup': 100,
            #     'beta_init': 0e-5,
            #     'beta': 0e-5,
            # },
            "optimizer": {
                "name": "AdamW",
                "kwargs": {
                    "lr": 2e-4 * 1 * 3 * bs / 4,  # 2.5e-3 for 1 gpu,  # multiplied by number of gpus
                    "weight_decay": 1e-5 * 1 * 3 * bs / 4,
                },
            },
            "scheduler": {
                "name": "LambdaLR",
                "kwargs": {
                    "lr_lambda": lambda t: 1 - (t / n_epochs),  # linear
                    # 'lr_lambda': lambda t: (1 + math.cos(math.pi * t / n_epochs) * .99) / 2,  # cosine
                    # 'lr_lambda': lambda t: math.exp(-7 * (t / n_epochs) ** 2),  # exponential
                },
                # 'name': 'StepLR',
                # 'kwargs': {
                #     'step_size': 1024,
                #     'gamma': 0.8,
                # }
            },
        },
        "random_crop": True,
    }

    dm = QGDataModule(**hparams["dm"])
    model = SdaLitModule(**hparams)

    metrics = {"loss": "avg. valid loss"}

    from datetime import datetime

    experiment_name = os.path.basename(__file__)[:-3]
    mlflow_logger = MLFlowLogger(
        experiment_name="qg3-" + experiment_name, run_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    trainer = L.Trainer(
        # devices=-1,
        # strategy=DDPStrategy(static_graph=True),
        max_epochs=n_epochs,
        logger=mlflow_logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=[
            # Plotter(),
            # EMA(decay=.99),
            # FreeGeneration(1),
            ModelCheckpoint(
                monitor="avg. valid loss",
                mode="min",
                save_top_k=2,
                save_last=True,
                dirpath=f"./ckpts/qg3/{experiment_name}",
            ),
        ],
        limit_val_batches=100,
    )

    trainer.fit(
        model=model,
        datamodule=dm,
        # ckpt_path=f'./ckpts/qg3/{experiment_name}/last.ckpt'
    )

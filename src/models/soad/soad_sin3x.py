import os
import sys

import lightning as L
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datasets.qg3.datamodule_ffcv import QGDataModule
from diffusion.scheduler import VPSDEScheduler

# from datasets.qg.datamodule_sda import QGDataModuleSDA
from modules.unet.unet_sda_seq2d import UNetSDA
from observations import get_observation


class FreeGeneration(L.Callback):
    def __init__(self, plot_every_nepoch: int = 10, num_inference_steps: int = 1000) -> None:
        super().__init__()

        self.plot_every_nepoch = plot_every_nepoch
        self.num_inference_steps = num_inference_steps

    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch % self.plot_every_nepoch:
            return

        scheduler = pl_module.diffusion_scheduler
        x0 = scheduler.sample(
            x1=torch.randn(1, 5, 4, 256, 256).cuda(),
            noise_estimator=pl_module.noise_estimator,
            n_steps=256,
            n_corrections=1,
            tau=0.25,
        )[0]
        # print(f'{z0.shape=}')# (1, 5, 16, 32, 32)

        plt.clf()

        fig, axs = plt.subplots(4, 5, figsize=(20, 16))

        for i in range(4):
            for j in range(5):
                axs[i, j].imshow(x0[j, i].cpu().numpy(), cmap="twilight", vmin=-3, vmax=3)
                axs[i, j].axis("off")
        plt.tight_layout()
        # fig.colorbar(im, ax=grid, orientation='horizontal')
        plt.close(fig)
        pl_module.logger.experiment.log_figure(
            run_id=pl_module.logger.run_id,
            figure=fig,
            artifact_file=f"figs/epoch={pl_module.current_epoch}_free_generation.png",
        )


class UNetSDAModule(L.LightningModule):
    def __init__(self, arch: dict, opt: dict, random_crop: bool = True, **kwargs) -> None:
        super().__init__()

        self.arch = arch
        self.opt = opt
        self.random_crop = random_crop

        self.noise_estimator = UNetSDA(**arch)
        self.diffusion_scheduler = VPSDEScheduler()

        self.opH = get_observation("sinusoidal", freq=3, sigma=0.01)

        if self.local_rank == 0:
            print(self.noise_estimator)
            print(self.diffusion_scheduler)

        self.valid_losses = []

        self.save_hyperparameters(dict(arch=arch, opt=opt, **kwargs))

    def configure_optimizers(self):
        optimizer = getattr(optim, self.opt["optimizer"]["name"])(
            self.noise_estimator.parameters(), **self.opt["optimizer"]["kwargs"]
        )
        if "scheduler" in self.opt:
            scheduler = getattr(optim.lr_scheduler, self.opt["scheduler"]["name"])(
                optimizer, **self.opt["scheduler"]["kwargs"]
            )
            return [optimizer], [scheduler]

        return optimizer

    def training_step(self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x = batch[0]
        b, L, *_ = x.shape
        # x.shape = (b, L, c, h, w)

        x = torch.cat([x, self.opH.func(x)], dim=-3)

        # if self.random_crop:
        #     x = torch.roll(x, shifts=torch.randint(x.shape[-1], size=(1,)).item(), dims=-1)
        #     x = torch.roll(x, shifts=torch.randint(x.shape[-2], size=(1,)).item(), dims=-2)

        # ? diffusion training
        t = self.diffusion_scheduler.sample_timesteps(x.shape[0]).to(x.device)
        xt, noise = self.diffusion_scheduler.add_noise(x, t)
        noise_pred = self.noise_estimator(xt, t)

        loss_l2 = F.mse_loss(noise_pred, noise)
        loss_l1 = (noise_pred - noise).abs().mean()  # L1 noise

        self.log("train_loss_l1", loss_l1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss_l2", loss_l2, on_epoch=True, sync_dist=True)

        return loss_l2

    def validation_step(self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x = batch[0]
        b, L, *_ = x.shape
        # x.shape = (b, L, c, h, w)

        x = torch.cat([x, self.opH.func(x)], dim=-3)

        # ? diffusion training
        t = self.diffusion_scheduler.sample_timesteps(x.shape[0]).to(x.device)
        xt, noise = self.diffusion_scheduler.add_noise(x, t)
        noise_pred = self.noise_estimator(xt, t)

        loss_l2 = F.mse_loss(noise_pred, noise)
        loss_l1 = (noise_pred - noise).abs().mean()  # L1 noise

        self.log("valid_loss_l1", loss_l1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss_l2", loss_l2, on_epoch=True, sync_dist=True)

        self.valid_losses.append(loss_l1)

        return loss_l2

    def on_validation_epoch_end(self) -> None:
        avg_valid_loss = torch.stack(self.valid_losses).mean()
        self.log("avg. valid loss", avg_valid_loss, sync_dist=True)
        self.valid_losses.clear()


if __name__ == "__main__":
    seed_everything(2357)

    n_epochs = 1024

    bs = 6

    hparams = {
        "dm": {
            "batch_size": bs,
            "seq_length": 9,
            "random_crop": True,
            "num_workers": 4,
        },
        "arch": {
            "channels": 4,
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
                    "lr": 5e-5 * 1 * 3 * bs / 4,  # 2.5e-3 for 1 gpu,  # multiplied by number of gpus
                    "weight_decay": 1e-6 * 1 * 3 * bs / 4,
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
    model = UNetSDAModule(**hparams)

    metrics = {"loss": "avg. valid loss"}

    from datetime import datetime

    experiment_name = os.path.basename(__file__)[:-3]
    mlflow_logger = MLFlowLogger(
        experiment_name="qg3-" + experiment_name, run_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    trainer = L.Trainer(
        # devices=num_gpus,
        # strategy=DDPStrategy(static_graph=True),
        max_epochs=n_epochs,
        logger=mlflow_logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=[
            # Plotter(),
            # EMA(decay=.99),
            FreeGeneration(1),
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
        # ckpt_path='last',
    )

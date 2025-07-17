import lightning as L
import torch
from lightning.pytorch.loggers import MLFlowLogger
from matplotlib import pyplot as plt


class FreeGeneration(L.Callback):
    def __init__(self, num_channels: int, plot_every_nepoch: int = 10, num_inference_steps: int = 1000) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.plot_every_nepoch = plot_every_nepoch
        self.num_inference_steps = num_inference_steps

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch % self.plot_every_nepoch:
            return

        scheduler = pl_module.diffusion_scheduler
        x0 = scheduler.sample(
            x1=torch.randn(1, 5, self.num_channels, 256, 256).cuda(),
            noise_estimator=pl_module.noise_estimator,
            n_steps=256,
            n_corrections=1,
            tau=0.25,
        )[0]
        # print(f'{z0.shape=}')# (1, 5, 16, 32, 32)

        plt.clf()

        fig, axs = plt.subplots(self.num_channels, 5, figsize=(20, self.num_channels * 4))

        for i in range(self.num_channels):
            for j in range(5):
                axs[i, j].imshow(x0[j, i].cpu().numpy(), cmap="twilight", vmin=-3, vmax=3)
                axs[i, j].axis("off")
        plt.tight_layout()
        # fig.colorbar(im, ax=grid, orientation='horizontal')
        plt.close(fig)

        for logger in pl_module.loggers:
            if isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=fig,
                    artifact_file=f"figs/epoch={pl_module.current_epoch}_free_generation.png",
                )

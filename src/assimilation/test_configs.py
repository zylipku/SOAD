import math
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import torch
from lightning import seed_everything

from ..data.qg_datamodule import QGDataModule
from ..models.components.unet.unet_sda_seq2d import UNetSDA
from ..observations import Arctan, Observation, Sinusoidal, Vor2Vel


class TestCaseConfig:
    datamodule: str
    observation: Observation
    extra_channels: int

    ckpt_path: Path

    ds_name_to_dm = {"qg": QGDataModule}
    obs_name_to_cls = {
        "arctan3x": partial(Arctan, scale=3.0),
        "sin3x": partial(Sinusoidal, freq=3.0),
        "vor2vel": Vor2Vel,
    }

    config_pool = {
        "sda": {
            "obs_type2channel": {
                "arctan3x": [0, 1],
                "sin3x": [0, 1],
                "vor2vel": [0, 1],
            },
            "ckpt_path": Path("ckpts/sda.ckpt"),
            "model": UNetSDA(channels=2),
        },
        "soad-arctan3x": {
            "obs_type2channel": {
                "arctan3x": [2, 3],
            },
            "ckpt_path": Path("ckpts/soad-arctan3x.ckpt"),
            "model": UNetSDA(channels=4),
        },
        "soad-sin3x": {
            "obs_type2channel": {
                "sin3x": [2, 3],
            },
            "ckpt_path": Path("ckpts/soad-sin3x.ckpt"),
            "model": UNetSDA(channels=4),
        },
        "soad-vor2vel": {
            "obs_type2channel": {
                "vor2vel": [2, 3, 4, 5],
            },
            "ckpt_path": Path("ckpts/soad-vor2vel.ckpt"),
            "model": UNetSDA(channels=6),
        },
        "soad-full": {
            "obs_type2channel": {
                "arctan3x": [2, 3],
                "sin3x": [4, 5],
                "vor2vel": [6, 7, 8, 9],
            },
            "ckpt_path": Path("ckpts/soad-full.ckpt"),
            "model": UNetSDA(channels=10),
        },
    }

    def __init__(
        self,
        ds_name: str,
        obs_types: Sequence[str] | str,
        rand_mask_ratio: float = None,
        rand_mask_isotropic: bool = False,
        subsample_stride: int = None,
        time_stride: Sequence[int] | int = 1,
        forecast_warmup_T: int = 1,
        # data assimilation
        method: str = "sda",
        seedno: int = 0,
        seq_length: int = 9,
    ):
        seed_everything(seedno)

        # dataset
        self.datamodule = self.ds_name_to_dm[ds_name](batch_size=1, seq_length=seq_length)
        if isinstance(obs_types, str):
            obs_types = [obs_types]

        if method == "maa-full" or method == "sda":
            cfg = self.config_pool[method]
        else:
            cfg = self.config_pool[method + "-" + obs_types[0]]

        # observation
        self.obs_channel_idxs = [cfg["obs_type2channel"][obs] for obs in obs_types]
        self.obs_nchannels = sum(len(channel_idxs) for channel_idxs in self.obs_channel_idxs)

        # rand_mask
        # rand_mask = None
        # if rand_mask_ratio is not None:
        #     assert 0 < rand_mask_ratio <= 1
        #     total_ngrids = math.prod(self.datamodule.snapshot_shape)
        #     num_obs = int(rand_mask_ratio * total_ngrids)
        #     obs_idx = torch.randperm(total_ngrids)[:num_obs]
        #     rand_mask = torch.zeros(total_ngrids, dtype=torch.bool)
        #     rand_mask[obs_idx] = True
        #     rand_mask = rand_mask.reshape(self.datamodule.snapshot_shape)
        # elif subsample_stride is not None:
        #     rand_mask = torch.zeros(self.datamodule.snapshot_shape, dtype=torch.bool)
        #     rand_mask[..., ::subsample_stride, ::subsample_stride] = True
        # else:
        #     rand_mask = torch.ones(self.datamodule.snapshot_shape, dtype=torch.bool)

        if rand_mask_ratio is not None:
            if rand_mask_isotropic:
                rand_masks = [
                    self._get_rand_mask(rand_mask_ratio),
                ] * len(obs_types)
            else:
                rand_masks = [self._get_rand_mask(rand_mask_ratio) for _ in range(len(obs_types))]

        elif subsample_stride is not None:
            rand_mask = torch.zeros(self.datamodule.snapshot_shape, dtype=torch.bool)
            rand_mask[..., ::subsample_stride, ::subsample_stride] = True
            rand_masks = [
                rand_mask,
            ] * len(obs_types)
        else:
            rand_mask = torch.ones(self.datamodule.snapshot_shape, dtype=torch.bool)
            rand_masks = [
                rand_mask,
            ] * len(obs_types)

        self.observations = [self.obs_name_to_cls[obs_type]() for obs_type in obs_types]
        self.rand_masks = rand_masks

        # model
        # assert method in ['sda', 'maa', 'maa-full']
        self.method = method
        self.model: UNetSDA = cfg["model"]
        # load ckpt
        self.ckpt_path = cfg["ckpt_path"]
        d = torch.load(self.ckpt_path)["state_dict"]
        self.model.load_state_dict({k.replace("noise_estimator.", ""): v for k, v in d.items()})

        self.seedno = seedno

        # time stride
        if isinstance(time_stride, int):
            self.time_strides = [
                time_stride,
            ] * len(obs_types)
        else:
            self.time_strides = list(time_stride)

        self.forecast_warmup_T = forecast_warmup_T

    def _get_rand_mask(self, ratio) -> torch.Tensor:
        assert 0 < ratio <= 1
        total_ngrids = math.prod(self.datamodule.snapshot_shape)
        num_obs = int(ratio * total_ngrids)
        obs_idx = torch.randperm(total_ngrids)[:num_obs]
        rand_mask = torch.zeros(total_ngrids, dtype=torch.bool)
        rand_mask[obs_idx] = True
        rand_mask = rand_mask.reshape(self.datamodule.snapshot_shape)
        return rand_mask

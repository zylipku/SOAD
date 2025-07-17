from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import lightning as L
import numpy as np
import torch
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.transforms import ToDevice, ToTensor

NORMALIZE_MEAN = None
NORMALIZE_STD = np.array([9.7e-6, 1.19e-6])


class RandomTranslate(Operation):
    def __init__(self, enabled: bool = False, translate_dims=(-2, -1)) -> None:
        super().__init__()

        self.enabled = enabled
        self.translate_dims = translate_dims

    def generate_code(self) -> Callable:
        def translate(img: np.ndarray, dst: np.ndarray) -> np.ndarray:
            dst = img
            if not self.enabled:
                return dst
            for dim in self.translate_dims:
                dst = np.roll(dst, shift=np.random.randint(0, dst.shape[dim]), axis=dim)
            return dst

        # translate.is_parallel = True
        return translate

    def declare_state_and_memory(self, previous_state: State) -> tuple[State, AllocationQuery | None]:
        return (replace(previous_state, jit_mode=False), AllocationQuery(previous_state.shape, previous_state.dtype))


class RandomWindow(Operation):
    def __init__(
        self,
        seq_length: int = 0,
        # -1 for full trajectory (-1, max_length, c, h, w)
        # 0 for single frame (-1, c, h, w)
    ) -> None:
        super().__init__()

        self.seq_length = seq_length

    def generate_code(self) -> Callable:
        seq_length = self.seq_length

        def crop(img: np.ndarray, dst: np.ndarray) -> np.ndarray:
            total_length = img.shape[-4]

            if seq_length == 0:
                rnd_idx = np.random.randint(0, total_length)
                dst = img[:, rnd_idx]
            elif seq_length == -1:
                dst = img
            else:
                rnd_idx = np.random.randint(0, total_length - seq_length + 1)
                dst = img[:, rnd_idx : rnd_idx + seq_length]
            return dst

        return crop

    def declare_state_and_memory(self, previous_state: State) -> tuple[State, AllocationQuery | None]:
        L, c, h, w = previous_state.shape

        if self.seq_length == 0:
            return (
                replace(previous_state, jit_mode=False, shape=(c, h, w)),
                AllocationQuery((c, h, w), previous_state.dtype),
            )
        elif self.seq_length == -1:
            return (
                replace(previous_state, jit_mode=False),
                AllocationQuery(previous_state.shape, previous_state.dtype),
            )
        else:
            return (
                replace(previous_state, jit_mode=False, shape=(self.seq_length, c, h, w)),
                AllocationQuery((self.seq_length, c, h, w), previous_state.dtype),
            )


class Normalize(Operation):
    def __init__(self, dim: int = -1, mean: torch.Tensor = None, std: torch.Tensor = None) -> None:
        super().__init__()

        self.dim = dim
        self.mean = mean
        self.std = std

    def generate_code(self) -> Callable:
        dim = self.dim
        mean = self.mean
        std = self.std

        def norm(img: np.ndarray, dst: np.ndarray) -> np.ndarray:
            dst = img.swapaxes(dim, -1)

            if mean is not None:
                dst -= mean
            if std is not None:
                dst /= std

            dst = dst.swapaxes(dim, -1)

            return dst

        return norm

    def declare_state_and_memory(self, previous_state: State) -> tuple[State, AllocationQuery | None]:
        return (replace(previous_state, jit_mode=False), AllocationQuery(previous_state.shape, previous_state.dtype))


class QGDataModule(L.LightningDataModule):
    ROOT_DIR = Path("/data/datasets/qg3")

    snapshot_shape = (256, 256)

    def __init__(
        self,
        batch_size: int = 64,
        seq_length: int = 9,
        shuffle: bool = False,
        random_crop: bool = False,
        num_workers: int = 4,
    ) -> None:
        """
        Lightning DataModule for the QG dataset.

        one batch consists of (x, kwargs), where
        `x.shape=(-1, 9, 6, 256, 256)` and `kwargs={}`

        Args:
            batch_size (int, optional): _description_. Defaults to 64.
            window_size (int, optional): _description_. Defaults to 9.
            shuffle (bool, optional): _description_. Defaults to False.
            num_workers (int, optional): _description_. Defaults to 4.
        """

        super().__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.random_crop = random_crop

        self.train_write_path = self.ROOT_DIR / "ffcv/train.beton"
        self.valid_write_path = self.ROOT_DIR / "ffcv/valid.beton"
        self.test_write_path = self.ROOT_DIR / "ffcv/test.beton"

        self.pipelines = {
            "traj": [
                RandomWindow(seq_length=self.seq_length),
                RandomTranslate(enabled=self.random_crop),
                Normalize(dim=-3, mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                ToTensor(),
            ]
        }

    def train_dataloader(self):
        loader = Loader(
            self.train_write_path,
            batch_size=self.batch_size,
            order=OrderOption.SEQUENTIAL,
            num_workers=self.num_workers,
            distributed=True,
            pipelines={
                "traj": [
                    RandomWindow(seq_length=self.seq_length),
                    RandomTranslate(enabled=self.random_crop),
                    Normalize(dim=-3, mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                    ToTensor(),
                    ToDevice(torch.device("cuda", self.trainer.local_rank)),
                ]
            },
        )
        return loader

    def val_dataloader(self):
        loader = Loader(
            self.valid_write_path,
            batch_size=self.batch_size,
            order=OrderOption.SEQUENTIAL,
            num_workers=self.num_workers,
            distributed=True,
            pipelines={
                "traj": [
                    RandomWindow(seq_length=self.seq_length),
                    Normalize(dim=-3, mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                    ToTensor(),
                    ToDevice(torch.device("cuda", self.trainer.local_rank)),
                ]
            },
        )
        return loader

    def test_dataloader(self):
        loader = Loader(
            self.test_write_path,
            batch_size=self.batch_size,
            order=OrderOption.SEQUENTIAL,
            num_workers=self.num_workers,
            pipelines={
                "traj": [
                    RandomWindow(seq_length=self.seq_length),
                    Normalize(dim=-3, mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                    ToTensor(),
                ]
            },
        )
        return loader

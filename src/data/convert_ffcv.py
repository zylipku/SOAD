import os
import sys
from pathlib import Path

import numpy as np
import torch
from ffcv.fields import NDArrayField
from ffcv.writer import DatasetWriter
from torch.nn import functional as F
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from constants import DATA_ROOT


class QGDataset(Dataset):
    def __init__(self, file_root: Path) -> None:
        super().__init__()

        file_names = os.listdir(file_root)
        self.file_names = [name for name in file_names if name.endswith(".npy")]
        self.file_paths = [file_root / name for name in self.file_names]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = np.load(self.file_paths[idx])
        x = F.avg_pool2d(torch.from_numpy(x).float(), 2).numpy()
        return (x,)


if __name__ == "__main__":
    train_dataset = QGDataset(DATA_ROOT / "qg3/train")
    valid_dataset = QGDataset(DATA_ROOT / "qg3/valid")
    test_dataset = QGDataset(DATA_ROOT / "qg3/test")

    fields = {
        "traj": NDArrayField(dtype=np.dtype("float32"), shape=(32, 2, 256, 256)),
    }

    train_writer = DatasetWriter(DATA_ROOT / "qg3/ffcv/train.beton", fields=fields, page_size=1 << 24, num_workers=-1)
    valid_writer = DatasetWriter(DATA_ROOT / "qg3/ffcv/valid.beton", fields=fields, page_size=1 << 24, num_workers=-1)
    test_writer = DatasetWriter(DATA_ROOT / "qg3/ffcv/test.beton", fields=fields, page_size=1 << 24, num_workers=-1)

    train_writer.from_indexed_dataset(train_dataset)
    valid_writer.from_indexed_dataset(valid_dataset)
    test_writer.from_indexed_dataset(test_dataset)

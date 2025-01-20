import glob
import os
import torch
import numpy as np
from typing import Optional
from torch.utils.data import Dataset
from utils import gen_hash
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class RaysDataset(Dataset):
    """
    Class captures an entire ray onto a fixed grid.
    Each iteration returns a grid of rays.
    """
    def __init__(self: 'RaysDataset', root_path: Optional[str] = None, name: Optional[str] = None) -> None:
        self.name = gen_hash() if name is None else name
        self.root_path = root_path
        self.mirror_dirs = list(set(glob.glob(os.path.join(self.root_path, "*"))))
        self.data_in = []
        self.data_out = []
        for mirror_dir in self.mirror_dirs:
            if os.path.isdir(mirror_dir):
                # 1. reading mirror
                mirror_path = os.path.join(mirror_dir, "mirror.txt")
                with open(mirror_path) as f:
                    mirror_name = f.read()
                    mirror = torch.load(mirror_name)
                    mirror = [torch.tensor(m) for m in mirror]
                input_dir_path = os.path.join(mirror_dir, "inputs")
                output_dir_path = os.path.join(mirror_dir, "outputs")
                # 2. loading any input-output pairs
                for i in range(len(os.listdir(input_dir_path))):
                    file_in = torch.from_numpy(np.load(os.path.join(input_dir_path, f"{i}.npy")))
                    file_out = torch.from_numpy(np.load(os.path.join(output_dir_path, f"{i}.npy")))
                    self.data_in.append((os.path.basename(mirror_dir), mirror, file_in.permute(2, 0, 1)))
                    self.data_out.append(file_out.permute(2, 0, 1))

        if len(self.data_in) == 0:
            raise ValueError("dataset is empty/corrupted.")

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self: 'RaysDataset', item: torch.Tensor) -> [tuple, tuple]:
        return self.data_in[item], self.data_out[item]


class BeamsDataset(Dataset):
    """
    Class captures an entire ray onto a fixed grid.
    Each iteration returns a grid of rays.
    """
    def __init__(self: 'BeamsDataset',
                 beams_in: Optional[list] = None,
                 beams_out: Optional[list] = None,
                 name: Optional[str] = None) -> None:
        self.name = gen_hash() if name is None else name
        self.data_in = [torch.from_numpy(bi).permute(2, 0, 1) for bi in beams_in]
        self.data_out = [torch.from_numpy(bo).permute(2, 0, 1) for bo in beams_out]

        if len(self.data_in) == 0:
            raise ValueError("dataset is empty/corrupted.")

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self: 'RaysDataset', item: torch.Tensor) -> [tuple, tuple]:
        return self.data_in[item], self.data_out[item]
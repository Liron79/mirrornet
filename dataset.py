import glob
import os
import torch
import numpy as np
from typing import Optional
from torch.utils.data import Dataset
from utils import gen_hash


class RaysDataset(Dataset):
    """
    Class captures an entire ray onto a fixed grid.
    Each iteration returns a grid of rays.
    """
    def __init__(self: 'RaysDataset', root_path: Optional[str] = None, name: Optional[str] = None) -> None:
        self.name = gen_hash() if name is None else name
        self.root_path = root_path
        mirror_dirs = list(set(glob.glob(os.path.join(self.root_path, "*"))))
        self.data = []
        for mirror_dir in mirror_dirs:
            if os.path.isdir(mirror_dir):
                # 1. reading mirror
                with open(os.path.join(mirror_dir, "mirror.txt")) as f:
                    mirror = torch.load(f.read())
                input_dir_path = os.path.join(mirror_dir, "inputs")
                output_dir_path = os.path.join(mirror_dir, "outputs")
                # 2. loading any input-output pairs
                for i in range(len(os.listdir(input_dir_path))):
                    file_in = torch.from_numpy(np.load(os.path.join(input_dir_path, f"{i}.npy")))
                    file_out = torch.from_numpy(np.load(os.path.join(output_dir_path, f"{i}.npy")))
                    data = torch.concat((file_in, file_out), dim=2).permute(2, 0, 1)
                    self.data.append((os.path.basename(mirror_dir), mirror, data, i))
                    # break

        if len(self.data) == 0:
            raise ValueError("dataset is empty.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self: 'RaysDataset', item: torch.Tensor) -> [tuple, tuple]:
        return self.data[item]
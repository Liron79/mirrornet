import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from torch import nn
import torch.cuda


def cuda(tensor: [nn.Module, torch.Tensor]) -> [nn.Module, torch.Tensor]:
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def load_data(path: str) -> Tuple[np.array, np.array, np.array]:
    if path is None or not os.path.exists(path):
        raise ValueError(f"path={path} is unexpected.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_rays(path: str) -> list:
    if path is None or not os.path.exists(path):
        raise ValueError(f"path={path} is unexpected.")
    rays = list()
    with open(path, newline='\n') as csvfile:
        next(csvfile)  # skip the first row which is considered as our headers
        for i, line in enumerate(csvfile):
            rays_tmp = list()
            for item in line.split(","):
                try:
                    rays_tmp.append(float(item))
                except Exception as _:
                    pass
            rays.append(rays_tmp)
    return rays


def physical_rays_drop(Ri: list, Ro: list) -> list:
    valid_idx = [line[-1] for line in Ro]
    Ri_new = list()
    for line in Ri:
        if line[-1] in valid_idx:
            Ri_new.append(line)
    return list(Ri_new)


def generate_physical_data(Ri: list, Ro: list, mirror_name: str) -> pd.DataFrame:
    Ri_columns = ["Ri_x", "Ri_y", "Ri_z",
                  "Ri_kx", "Ri_ky", "Ri_kz",
                  "Ri_ex", "Ri_ey", "Ri_ez",
                  "Ri_distance", "Ri_amp", "Ri_status",
                  "Ri_ray_index"]
    Ro_columns = ["Ro_x", "Ro_y", "Ro_z",
                  "Ro_kx", "Ro_ky", "Ro_kz",
                  "Ro_ex", "Ro_ey", "Ro_ez",
                  "Ro_distance", "Ro_amp", "Ro_status",
                  "Ro_ray_index"]
    data = pd.DataFrame(columns=["M"] + Ri_columns + Ro_columns)
    for i in range(len(Ri)):
        if not isinstance(Ri[i], list):
            Ri[i] = Ri[i].tolist()
        if not isinstance(Ri[i], list):
            Ro[i] = Ro[i].tolist()
        data.loc[i] = [mirror_name] + Ri[i] + Ro[i]
    return data


def transform(Ro: list, kind: str = "linear") -> list:
    for i, ro in enumerate(Ro):
        x, y, z, kx, ky, kz, ex, ey, ez, distance, amp, status, ray_index = ro
        if kind == "linear":
            x = x + 110 - 40
            z = 110 + 55 - 40
            kx = kx * -1
            kz = kz * -1
        Ro[i] = [x, y, z, kx, ky, kz, ex, ey, ez, distance, amp, status, ray_index]

    return Ro

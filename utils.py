import hashlib
import os
import pickle
import random
import json
import numpy as np
import pandas as pd
import torch.cuda
from typing import Tuple
from torch import nn
from datetime import datetime


def current_time():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = datetime.fromtimestamp(timestamp)
    return dt_object


def gen_hash(size=10):
    return hashlib.sha1(str(random.getrandbits(256)).encode('utf-8')).hexdigest()[:size]


def cuda(tensor: [nn.Module, torch.Tensor]) -> [nn.Module, torch.Tensor]:
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def load_json(path: str) -> dict:
    if path is None or not os.path.exists(path):
        raise ValueError(f"path={path} is unexpected.")
    with open(path, "rb") as f:
        data = json.load(f)
    return data


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


def physical_rays_drop(Ri: list, Ro: list, Tkxi: float, Tkyi: float) -> Tuple[list, list]:
    valid_idx = [line[-1] for line in Ro]
    Ri_ = list()
    for line in Ri:
        if line[-1] in valid_idx:
            Ri_.append(line)

    Ri_new = list()
    Ro_new = list()
    for ri, ro in zip(Ri_, Ro):
        kxi = abs(ri[3])
        kyi = abs(ri[4])
        if kxi > Tkxi or kyi > Tkyi:
            continue
        Ri_new.append(ri)
        Ro_new.append(ro)

    return list(Ri_new), list(Ro_new)


def generate_physical_data(Ri: list, Rint: list, Ro: list, mirror_name: str) -> pd.DataFrame:
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
    Rint_columns = ["Rint_x", "Rint_y", "Rint_z",
                    "Rint_kx", "Rint_ky", "Rint_kz",
                    "Rint_ex", "Rint_ey", "Rint_ez",
                    "Rint_distance", "Rint_amp", "Rint_status",
                    "Rint_ray_index"]
    data_M = pd.DataFrame(columns=["M"], data=[mirror_name] * len(Ri))
    data_Ri = pd.DataFrame(Ri, columns=Ri_columns)
    data_Ro = pd.DataFrame(Ro, columns=Ro_columns)
    data_Rint = pd.DataFrame(Rint, columns=Rint_columns)
    return pd.concat((data_M, data_Ri, data_Ro, data_Rint), axis=1)


def minmax_norm(t, a, b, t_max, t_min):
    return a + (((t - t_min) * (b - a)) / (t_max - t_min))


def generate_rays_dataset(Ri: list, Ro: list,
                          x_dim: int, y_dim: int,
                          xi_min: int, xi_max: int,
                          yi_min: int, yi_max: int):
    R = dict()
    for ri, ro in zip(Ri, Ro):
        xi, yi, _, kxi, kyi, _, _, _, _, _, _, _, _ = ri
        _, _, _, _, kyo, kzo, _, _, _, _, _, _, _ = ro

        xi = minmax_norm(xi, 0, x_dim - 1, xi_max, xi_min)
        xi = int(xi)
        yi = minmax_norm(yi, 0, y_dim - 1, yi_max, yi_min)
        yi = int(yi)

        idx = (xi, yi)
        values = (kxi, kyi, kyo, kzo)
        if idx not in R:
            R[idx] = [values]
        else:
            R[idx].append(values)

    max_len = 30
    for idx in R:
        R[idx] = sorted(R[idx], key=lambda pair: (pair[0] ** 2 + pair[1] ** 2) ** 0.5)
        # max_len = max(max_len, len(R[idx]))
    print(f"max_len =  {max_len}")

    R_input = list()
    R_output = list()
    par_mode = True
    for _ in range(max_len):
        Ri_kx_mat = np.ones((x_dim, y_dim, 1))
        Ri_ky_mat = np.ones((x_dim, y_dim, 1))
        Ro_ky_mat = np.ones((x_dim, y_dim, 1))
        Ro_kz_mat = np.ones((x_dim, y_dim, 1))
        for idx in R:
            x, y = idx
            j = 0 if par_mode else random.randint(0, len(R[idx]) - 1)
            kxi, kyi, kyo, kzo = R[idx][j]

            Ri_kx_mat[x, y] = kxi
            Ri_ky_mat[x, y] = kyi
            Ro_ky_mat[x, y] = kyo
            Ro_kz_mat[x, y] = kzo

        par_mode = False
        R_input.append(np.concatenate((Ri_kx_mat, Ri_ky_mat), axis=2))
        R_output.append(np.concatenate((Ro_ky_mat, Ro_kz_mat), axis=2))

    return R_input, R_output


def transform(Ro: list, kind: str = "linear") -> list:
    for i, ro in enumerate(Ro):
        x, y, z, kx, ky, kz, ex, ey, ez, distance, amp, status, ray_index = ro
        if kind == "linear":
            # x = x + 110 - 40
            # z = 110 + 55 - 40
            kx = kx * -1
            kz = kz * -1
        Ro[i] = [x, y, z, kx, ky, kz, ex, ey, ez, distance, amp, status, ray_index]

    return Ro


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
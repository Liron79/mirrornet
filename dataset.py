import torch
import numpy as np
import pandas as pd
from typing import Optional
from torch.utils.data import Dataset
from utils import gen_hash


def cell_sampling(data: pd.DataFrame, cell_resolution: int) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Data is Empty.")
    sampled_data = list()
    for name, grp in data.groupby(by=["M", "Ri_x", "Ri_y", "Ri_z"]):
        # Parallel rays approach
        sampled_grp = grp.sort_values(by=["Ri_kx", "Ri_ky", "Ri_kz"], key=abs).reset_index(drop=True)
        # parael_rays_cond = (sampled_grp.Ri_kx == 0.0) & (sampled_grp.Ri_ky == 0.0) & (sampled_grp.Ri_kz == -1.0)
        # sampled_grp = sampled_grp[parael_rays_cond]
        sampled_grp = sampled_grp.iloc[:cell_resolution]
        sampled_data.append(sampled_grp)
    return pd.concat(sampled_data, axis=0)


class PhysicalDataset(Dataset):

    def __init__(self: 'PhysicalDataset', paths: list, name: Optional[str] = None, n_rows: Optional[int] = None, cell_resolution: int = 1) -> None:
        self.name = gen_hash() if name is None else name
        self.paths = paths
        self.cell_resolution = cell_resolution
        self.data = pd.concat([pd.read_csv(path, nrows=n_rows) for path in paths], axis=0).reset_index(drop=True)
        # self.sampled_data = pd.DataFrame(columns=self.data.columns)
        if self.data.empty:
            raise ValueError("dataset is empty.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self: 'PhysicalDataset', item: torch.Tensor) -> [tuple, tuple]:
        M = [torch.load(M_path) for M_path in [self.data.M.iloc[item]]]
        Mx = torch.cat([torch.tensor(Mx).float() for (Mx, _, _) in M], dim=0)
        My = torch.cat([torch.tensor(My).float() for (_, My, _) in M], dim=0)
        Mz = torch.cat([torch.tensor(Mz).float() for (_, _, Mz) in M], dim=0)
        M = (Mx, My, Mz)
        sep = len(self.data.columns[1:]) // 3
        data = self.data.iloc[item].to_frame().T
        Ri = torch.from_numpy(data[data.columns[1:sep + 1]].values.astype(np.float32))[0, ...]
        # Ri = Ri[:, :3] # x, y, z, kx, ky, kz
        Ro = torch.from_numpy(data[data.columns[sep + 1:sep*2 + 1]].values.astype(np.float32))[0, ...]
        # Ro = Ro[:, :3] # x, y, z, kx, ky, kz
        Rint = torch.from_numpy(data[data.columns[sep*2 + 1:]].values.astype(np.float32))[0, ...]
        R = torch.cat((Ri, Ro, Rint), dim=0).float()
        return R, Ri, Ro, Rint, M


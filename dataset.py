import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import gen_hash


class PhysicalDataset(Dataset):

    def __init__(self, paths: list) -> None:
        self.name = gen_hash()
        self.paths =  paths
        self.data = pd.concat([pd.read_csv(path) for path in paths], axis=0)
        if self.data.empty:
            raise ValueError("dataset is empty.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item) -> [tuple, tuple]:
        M = [torch.load(M_path) for M_path in [self.data.M.iloc[item]]]
        Mx = torch.cat([torch.tensor(Mx).float() for (Mx, _, _) in M], dim=0)
        My = torch.cat([torch.tensor(My).float() for (_, My, _) in M], dim=0)
        Mz = torch.cat([torch.tensor(Mz).float() for (_, _, Mz) in M], dim=0)
        M = (Mx, My, Mz)
        sep = len(self.data.columns[1:]) // 2
        Ri = torch.from_numpy(self.data[self.data.columns[1:sep + 1]].values)
        Ro = torch.from_numpy(self.data[self.data.columns[sep + 1:]].values)
        R = torch.cat((Ri, Ro), dim=1).float()[item]
        return R, M


import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from utils import cuda


def Zmirror_prediction(model: nn.Module, dataloader: DataLoader) -> np.array:
    final_z = list()
    for (R, Ri, Ro, _, M) in iter(dataloader):
        with torch.no_grad():
            Ri = cuda(Ri[:, :3])
            Ro = cuda(Ro[:, :3])
            R_ = torch.cat((Ri, Ro), dim=1).float()
            z = model(cuda(R_))
            final_z.append(z.detach().cpu())
    final_z = torch.cat(final_z, dim=0).mean(axis=0)

    return final_z.numpy()


class ZMirrorLoss(nn.Module):
    def __init__(self: 'ZMirrorLoss', reduction: str = "mean") -> None:
        super(ZMirrorLoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self: 'ZMirrorLoss', Mo: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        Lz = self.loss_fn(Mo, M)

        return Lz


class ZMirrorModel(nn.Module):
    def __init__(self: 'ZMirrorModel', in_features: int = 6, out_features: int = 625):
        super(ZMirrorModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features, 10, bias=True)
        self.fc2 = nn.Linear(10, out_features, bias=True)

    def forward(self: 'ZMirrorModel', R: torch.Tensor) -> tuple:
        # R = [3 parameters of Ri, 3 parameters of Ro] -> M = [625]
        Mo = self.fc1(R)
        Mo = self.relu(Mo)
        z = self.fc2(Mo)

        return z
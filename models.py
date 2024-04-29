import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import cuda


def mirror_prediction(model: nn.Module, dataloader: DataLoader) -> tuple:
    final_x = list()
    final_y = list()
    final_z = list()
    for R, M in iter(dataloader):
        with torch.no_grad():
            x, y, z = model(cuda(R))
            final_x.append(x.detach().cpu())
            final_y.append(y.detach().cpu())
            final_z.append(z.detach().cpu())
    final_x = torch.cat(final_x, dim=0).mean(axis=0)
    final_y = torch.cat(final_y, dim=0).mean(axis=0)
    final_z = torch.cat(final_z, dim=0).mean(axis=0)

    return final_x.numpy(), final_y.numpy(), final_z.numpy()


class MirrorLoss(nn.Module):
    def __init__(self: 'MirrorLoss', reduction: str = "mean") -> None:
        super(MirrorLoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self: 'MirrorLoss', Mo: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        Lx = self.loss_fn(Mo[0], M[0])
        Ly = self.loss_fn(Mo[1], M[1])
        Lz = self.loss_fn(Mo[2], M[2])

        return (Lx + Ly + Lz) / 3.0


class MirrorModel(nn.Module):
    def __init__(self: 'MirrorModel'):
        super(MirrorModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(26, 256, bias=True)
        self.fc2 = nn.Linear(256, 25 + 25 + 625, bias=True)

    def forward(self: 'MirrorModel', R: torch.Tensor) -> tuple:
        # R = [13 parameters of Ri, 13 parameters of Ro] -> M = [25, 25, 625]
        Mo = self.fc1(R)
        Mo = self.relu(Mo)

        Mo = self.fc2(Mo)

        x = Mo[:, :25]
        y = Mo[:, 25:50]
        z = Mo[:, 50:]

        return x, y, z

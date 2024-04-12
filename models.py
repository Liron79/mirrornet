import torch
from torch import nn


def mirror_prediction(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple:
    final_x = list()
    final_y = list()
    final_z = list()
    for R, M in iter(dataloader):
        with torch.no_grad():
            x, y, z = model(R.cuda())
            final_x.append(x.detach().cpu())
            final_y.append(y.detach().cpu())
            final_z.append(z.detach().cpu())
    final_x = torch.cat(final_x, dim=0).mean(axis=0)
    final_y = torch.cat(final_y, dim=0).mean(axis=0)
    final_z = torch.cat(final_z, dim=0).mean(axis=0)

    return final_x.numpy(), final_y.numpy(), final_z.numpy()


class MirrorMSELoss(nn.Module):
    def __init__(self: 'MirrorMSELoss', reduction: str = "mean") -> None:
        super(MirrorMSELoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self: 'MirrorMSELoss', Mo: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        Lx = self.loss_fn(Mo[0], M[0])
        Ly = self.loss_fn(Mo[1], M[1])
        Lz = self.loss_fn(Mo[2], M[2])

        return (Lx + Ly + Lz) / 3.0


class MirrorModel(nn.Module):
    def __init__(self: 'MirrorModel'):
        super(MirrorModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(26, 512, bias=True)
        self.fc2 = nn.Linear(512, 1024, bias=True)
        self.fc3 = nn.Linear(1024, 675, bias=True)
        self.fc4 = nn.Linear(675, 25 + 25 + 625, bias=True)

    def forward(self: 'MirrorModel', R: torch.Tensor) -> tuple:
        # R = [13 parameters of Ri, 13 parameters of Ro] -> M = [25, 25, 625]
        Mo = self.fc1(R)
        Mo = self.relu(Mo)

        Mo = self.fc2(Mo)
        Mo = self.relu(Mo)

        Mo = self.fc3(Mo)
        Mo = self.relu(Mo)

        Mo = self.fc4(Mo)

        x = Mo[:, :25]
        y = Mo[:, 25:50]
        z = Mo[:, 50:]

        return x, y, z

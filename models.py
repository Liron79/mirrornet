import torch
from torch import nn


class ZMirrorLoss(nn.Module):
    """
    Generalizes loss concept between mirrors.
    """
    def __init__(self: 'ZMirrorLoss', reduction: str = "mean") -> None:
        super(ZMirrorLoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self: 'ZMirrorLoss', Mo: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        Lz = self.loss_fn(Mo, M)

        return Lz


class ZMirrorModel(nn.Module):
    """
    Neural network model which takes [Ri;Ro] and returns M (new mirror).
    """
    def __init__(self: 'ZMirrorModel', in_features: int = 400, out_features: int = 625):
        super(ZMirrorModel, self).__init__()
        self.act = nn.SiLU(inplace=True)
        self.fc1 = nn.Linear(in_features, 450, bias=True)
        self.fc2 = nn.Linear(450, 500, bias=True)
        self.fc3 = nn.Linear(500, 550, bias=True)
        self.fc4 = nn.Linear(550, 600, bias=True)
        self.fc5 = nn.Linear(600, out_features, bias=True)

    def forward(self: 'ZMirrorModel', R: torch.Tensor) -> tuple:
        Mo = self.fc1(R)
        Mo = self.act(Mo)
        Mo = self.fc2(Mo)
        Mo = self.act(Mo)
        Mo = self.fc3(Mo)
        Mo = self.act(Mo)
        Mo = self.fc4(Mo)
        Mo = self.act(Mo)
        z = self.fc5(Mo)

        return z
import torch
from torch import nn


class ZMirrorLoss(nn.Module):
    """
    Generalizes loss concept between mirrors.
    """
    def __init__(self: 'ZMirrorLoss', reduction: str = "mean") -> None:
        super(ZMirrorLoss, self).__init__()
        self.reduction = reduction
        # self.loss_fn = nn.MSELoss(reduction=reduction)
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self: 'ZMirrorLoss', Mo: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        Lz = self.loss_fn(Mo, M)

        return Lz


class ZMirrorModel(nn.Module):
    """
    Neural network model which takes [Ri;Ro] and returns M (new mirror).
    """
    def __init__(self: 'ZMirrorModel', in_features: int, in_output_features: int, out_features: int = 625):
        super(ZMirrorModel, self).__init__()
        self.act = nn.ReLU(inplace=True)
        d = 0.5
        hidden_neurons = int(d * 1024)
        self.fc1 = nn.Linear(in_features + in_output_features, hidden_neurons, bias=True)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons, bias=True)
        self.fc3 = nn.Linear(hidden_neurons, out_features, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_neurons)
        self.bn2 = nn.BatchNorm1d(hidden_neurons)

    def forward(self: 'ZMirrorModel', R: torch.Tensor, O: torch.Tensor) -> tuple:
        z = torch.concat((R, O), axis=1)
        z = self.fc1(z)
        z = self.bn1(z)
        z = self.act(z)
        z = self.fc2(z)
        z = self.bn2(z)
        z = self.act(z)
        z = self.fc3(z)

        return z
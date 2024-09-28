import os.path
import torch
import numpy as np
import pandas as pd
from typing import Optional, List
from matplotlib import pyplot as plt
from models import ZMirrorLoss


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mirror_dir_path = os.path.join(base_dir, "MirrorModels")
vis_data_path = os.path.join(base_dir, "Storage", "VisData")

path = os.path.join(mirror_dir_path, "2024_09_28_21_48_21", "epoch_metrics.csv")
pred_mirror_path = os.path.join(base_dir, "MirrorModels", "GeneratedMirrors", "2024_09_28_21_55_15", "super_comp_3gauss_thresh01_parabolicPFL0.5RFL40", "mirror.pt")
gt_mirror = os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL40.pt")


def loss_mirrors(gt_Z: np.array, pred_Z: np.array) -> list:
    L = list()
    zloss = ZMirrorLoss()
    L.append(zloss(torch.from_numpy(pred_Z), torch.from_numpy(gt_Z)).item())

    return L


def loss_curve(path: str, title: str = str(), start: int = 0, end: int = -1, save: Optional[str] = None) -> None:
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(df.Epoch_Loss[start:end])
    if save is not None and os.path.exists(save):
        plt.savefig(save)
    plt.show()


if __name__ == "__main__":
    print(f"Predicted mirror: {pred_mirror_path}")
    loss_curve(path)

    _, _, final_Z = torch.load(pred_mirror_path)
    _, _, gt_Z = torch.load(gt_mirror)
    L = loss_mirrors(gt_Z=gt_Z, pred_Z=final_Z)
    print(f"{gt_mirror}: {L=}")
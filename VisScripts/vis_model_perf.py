import os.path
import torch

import seaborn as sns
import numpy as np
import pandas as pd

from typing import Optional, List

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from models import ZMirrorLoss


def loss_mirrors(gt_Z_list: List[np.array], pred_Z: np.array) -> list:
    L = list()
    zloss = ZMirrorLoss()
    for gt_Z in gt_Z_list:
        L.append(zloss(torch.from_numpy(pred_Z), torch.from_numpy(gt_Z)).item())
    return L


def perf(gt_dfs: List[pd.DataFrame], pred_df: pd.DataFrame):
    Lx = list()
    Ly = list()
    Lz = list()

    for gt_df in gt_dfs:
        gt_df_ = gt_df.loc[gt_df.Ro_ray_index.isin(pred_df.Ro_ray_index.unique().tolist())]
        gt_df_ = gt_df_.sort_values(by="Ro_ray_index")
        pred_df_ = pred_df.sort_values(by="Ro_ray_index")

        Lx.append(abs(pred_df_.Ro_x.values - gt_df_.Ro_x.values).mean())
        Ly.append(abs(pred_df_.Ro_y.values - gt_df_.Ro_y.values).mean())
        Lz.append(abs(pred_df_.Ro_z.values - gt_df_.Ro_z.values).mean())

        df = pd.DataFrame()
        df['gZ'] = gt_df_.Ro_z.round(3)
        df['pZ'] = pred_df_.Ro_z.round(3)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        sns.regplot(x='pZ', y='gZ', data=df)
        plt.show()

    Lx = np.mean(Lx)
    Ly = np.mean(Ly)
    Lz = np.mean(Lz)

    return (Lx + Ly + Lz) / 3.0


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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mirror_dir_path = os.path.join(base_dir, "MirrorModels")
    vis_data_path = os.path.join(base_dir, "Storage", "VisData")
    mirror_name = "275172cbff"
    path = os.path.join(mirror_dir_path, mirror_name, "epoch_loss.csv")
    # loss_curve(path=path, title=mirror_name, start=0, end=-1, save=os.path.join(vis_data_path, f"{mirror_name}.png"))

    mirror_pred_path = os.path.join(base_dir, "MirrorModels", "275172cbff", "mirror.pt")
    _, _, final_Z = torch.load(mirror_pred_path)
    mirror_gt_paths = [
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL30.pt"),
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL35.pt"),
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL40.pt"),
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL45.pt"),
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL50.pt"),
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL55.pt"),
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL60.pt"),
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL90.pt")
    ]
    gt_z_list = [torch.load(p)[-1] for p in mirror_gt_paths]
    L = loss_mirrors(gt_Z_list=gt_z_list, pred_Z=final_Z)
    print("Training", f"{L=}")
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(range(len(L)), L)
    # ax[0].set_xticklabels([os.path.basename(x).split(".pt")[0].split("parabolicPFL0.5")[-1] for x in mirror_gt_paths])

    mirror_pred_path = os.path.join(base_dir, "MirrorModels", "9fa846e6e6", "mirror.pt")
    _, _, final_Z = torch.load(mirror_pred_path)
    mirror_gt_paths = [
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL38.pt"),
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL48.pt"),
        os.path.join(base_dir, "PhysicalMirrors", "parabolicPFL0.5RFL68.pt")
    ]
    gt_z_list = [torch.load(p)[-1] for p in mirror_gt_paths]
    L = loss_mirrors(gt_Z_list=gt_z_list, pred_Z=final_Z)
    print("Validation", f"{L=}")
    ax[1].plot(range(len(L)), L)
    # ax[1].set_xticklabels([os.path.basename(x).split(".pt")[0].split("parabolicPFL0.5")[-1] for x in mirror_gt_paths])
    plt.tight_layout()
    plt.show()

    ######
    # pred_path = os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_275172cbff_mirror.csv")
    # pred_df = pd.read_csv(pred_path)
    # gt_paths = [
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL30.csv"),
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL35.csv"),
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL40.csv"),
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL45.csv"),
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL50.csv"),
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL55.csv"),
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL60.csv"),
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL90.csv")
    # ]
    # gt_dfs = [pd.read_csv(p) for p in gt_paths]
    # P = perf(gt_dfs=gt_dfs, pred_df=pred_df)
    # print("Training", f"{P=}")
    #
    # pred_path = os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_9fa846e6e6_mirror.csv")
    # pred_df = pd.read_csv(pred_path)
    # gt_paths = [
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL38.csv"),
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL48.csv"),
    #     os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL68.csv")
    # ]
    # gt_dfs = [pd.read_csv(p) for p in gt_paths]
    # P = perf(gt_dfs=gt_dfs, pred_df=pred_df)
    # print("Validation", f"{P=}")
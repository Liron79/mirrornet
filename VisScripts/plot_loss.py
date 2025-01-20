import os.path
import pandas as pd
from typing import Optional
from matplotlib import pyplot as plt


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mirror_dir_path = os.path.join(base_dir, "MirrorModels")
vis_data_path = os.path.join(base_dir, "Storage", "VisData")

path = os.path.join(mirror_dir_path, "<model dir name>")


def loss_curve(path: str, title: str = str(), start: int = 0, end: int = -1, save: Optional[str] = None) -> None:
    if not os.path.exists(path):
        return

    df = pd.read_csv(os.path.join(path, "epoch_metrics.csv"))
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(df.Epoch_Loss[start:end])
    if save is not None and os.path.exists(save):
        plt.savefig(save)
    plt.show()


if __name__ == "__main__":
    print(f"Loading: {path}")
    loss_curve(path)
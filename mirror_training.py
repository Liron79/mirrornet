import torch
import os
import json
import pandas as pd
import numpy as np
from utils import cuda
from dataset import PhysicalDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models import MirrorMSELoss, MirrorModel, mirror_prediction

# python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

base_dir = os.path.dirname(os.path.abspath(__file__))
mirrors_dir = os.path.join(base_dir, "Mirrors")
os.makedirs(mirrors_dir, exist_ok=True)

dir_path = os.path.join(base_dir, "PhysicalData")
data_list = [
    os.path.join(dir_path, "pulse_1x1x1_parabolic.csv"),
]

print_every = 5
batch_size = 2
epochs = 2
lr = 0.01


if __name__ == "__main__":
    # 1. load Data from PhysicalData directory
    dataset = PhysicalDataset(paths=data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    metadata = {
        "input_paths": dataset.paths,
        "batch_size": batch_size,
        "lr_start": float(lr),
    }

    if torch.cuda.is_available():
        print("Training a model with CUDA GPU")
    else:
        print("Training a model with CPU")

    # 2. train a mirror model from the Data
    loss_fn = cuda(MirrorMSELoss())
    epoch_loss = np.zeros(epochs)
    model = cuda(MirrorModel())
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=len(dataloader) * epochs)

    for e in range(epochs):
        batch_loss = list()
        for (R, M) in iter(dataloader):
            Mo = model(cuda(R))
            Mo = [cuda(axis) for axis in Mo]
            M = [cuda(axis) for axis in M]
            loss = loss_fn(Mo, M)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            batch_loss.append(loss.detach().cpu().numpy())

        epoch_loss[e] = np.mean(batch_loss)
        if e % print_every == 0:
            print(f"Epoch={e},LR={lr},Averaged-Loss={epoch_loss[e]}")

    metadata["lr_end"] = float(lr)

    mirror_dir = os.path.join(mirrors_dir, dataset.name)
    os.makedirs(mirror_dir, exist_ok=True)

    df = pd.DataFrame(columns=["Epoch_Loss"])
    df.Epoch_Loss = epoch_loss.tolist()
    df.to_csv(os.path.join(mirror_dir, "epoch_loss.csv"), index=False)

    with open(os.path.join(mirror_dir, "metadata.json"), "w+") as f:
        json.dump(metadata, f, indent=4)
    torch.save(model.eval().cpu(), os.path.join(mirror_dir, "model.pt"))

    # 3. generate an averaged mirror and save it to Mirrors directory
    final_M = mirror_prediction(model=cuda(model), dataloader=dataloader)
    torch.save(final_M, os.path.join(mirror_dir, "mirror.pt"))

    print(f"Model results directory: {mirror_dir}")
    print("Done!")
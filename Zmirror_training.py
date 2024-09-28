import torch
import os
import json
import pandas as pd
import numpy as np
from utils import cuda, current_time, init_weights
from dataset import RaysDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models import ZMirrorLoss, ZMirrorModel

# python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

batch_size = 64
epochs = 400
lr = 0.0005
checkpoint_rate = 100

base_dir = os.path.dirname(os.path.abspath(__file__))
mirrors_dir = os.path.join(base_dir, "MirrorModels")
os.makedirs(mirrors_dir, exist_ok=True)

dir_path = os.path.join(base_dir, "Storage", "PhysicalData")
dataset_path = os.path.join(dir_path, "RaysMatrix", "train")


if __name__ == "__main__":
    dt_object = current_time()
    date = dt_object.strftime('%Y_%m_%d')
    time = dt_object.strftime('%H_%M_%S')
    print(f"Mirror Flow Start Time = {dt_object}")

    # mirror_dir = os.path.join(mirrors_dir, dataset.name)
    mirror_dir = os.path.join(mirrors_dir, date + "_" + time)
    os.makedirs(mirror_dir, exist_ok=True)

    # 1. load Data from PhysicalData directory
    dataset = RaysDataset(root_path=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    metadata = {
        "batch_size": batch_size,
        "lr_start": lr,
        "epochs": epochs
    }

    if torch.cuda.is_available():
        print("Training a model with CUDA GPU")
    else:
        print("Training a model with CPU")

    # 2. train a mirror model from the Data
    loss_fn = cuda(ZMirrorLoss())
    epoch_stdev = np.zeros(epochs)
    epoch_loss = np.zeros(epochs)
    model = cuda(ZMirrorModel(in_features=24576))
    model.apply(init_weights)
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    for e in range(epochs):
        batch_stdev = list()
        batch_loss = list()
        for _, M, data, i in iter(dataloader):
            Zo = cuda(model(cuda(torch.flatten(data, start_dim=1).float())))[0, ...]
            Z = M[2].float()[0, ...]
            loss = loss_fn(Zo, Z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stdev = ((Zo - Z).abs() / Z).mean().detach().cpu().numpy()
            batch_stdev.append(stdev)
            batch_loss.append(loss.detach().cpu().numpy())

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        epoch_stdev[e] = np.mean(batch_stdev)
        epoch_loss[e] = np.mean(batch_loss)
        if e % 5 == 0:
            print(f"Epoch={e}/{epochs},{lr=},{epoch_loss[e]=},{epoch_stdev[e]=}")

        # if e % checkpoint_rate == 0:
        #     checkpoint_path = os.path.join(mirrors_dir, mirror_dir, "checkpoint", f"{e}")
        #     os.makedirs(checkpoint_path, exist_ok=True)
        #     torch.save(model.eval().cpu(), os.path.join(checkpoint_path, "model.pt"))

    metadata["lr_end"] = lr

    df = pd.DataFrame(columns=["Epoch_Loss", "Epoch_Stdev"])
    df.Epoch_Stdev = epoch_stdev.tolist()
    df.Epoch_Loss = epoch_loss.tolist()
    df.to_csv(os.path.join(mirror_dir, "epoch_metrics.csv"), index=False)

    dt_object_end = current_time()
    print("Training End Time =", dt_object_end)
    training_time = (dt_object_end - dt_object).total_seconds() / 3600
    metadata["training_time_hr"] = training_time
    print("Training duration (Hours) =", training_time)

    with open(os.path.join(mirror_dir, "metadata.json"), "w+") as f:
        json.dump(metadata, f, indent=4)
    torch.save(model.eval().cpu(), os.path.join(mirror_dir, "model.pt"))

    print(f"Model results directory: {mirror_dir}")
    print("Done!")
import torch
import os
import json
import pandas as pd
import numpy as np
from utils import cuda, current_time, load_json, init_weights
from dataset import PhysicalDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models import ZMirrorLoss, ZMirrorModel, Zmirror_prediction

# python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

batch_size = 64
epochs = 200
lr = 0.01
checkpoint_rate = 20

base_dir = os.path.dirname(os.path.abspath(__file__))
mirrors_dir = os.path.join(base_dir, "MirrorModels")
os.makedirs(mirrors_dir, exist_ok=True)

dir_path = os.path.join(base_dir, "Storage", "PhysicalData")
data_list = load_json("training_cfg.json")["training_paths"]
data_list = [os.path.join(dir_path, p) for p in data_list]


if __name__ == "__main__":
    dt_object = current_time()
    print("Training Start Time =", dt_object)

    # 1. load Data from PhysicalData directory
    dataset = PhysicalDataset(paths=data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    metadata = {
        "input_paths": data_list,
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
    epoch_loss = np.zeros(epochs)
    model = cuda(ZMirrorModel())
    model.apply(init_weights)
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    final_X = None
    final_Y = None
    for e in range(epochs):
        batch_loss = list()
        for (R, Ri, Ro, _, M) in iter(dataloader):
            Ri = cuda(Ri[:, :3])
            Ro = cuda(Ro[:, :3])
            R_ = torch.cat((Ri, Ro), dim=1).float()
            Zo = cuda(model(cuda(R_)))
            Z = cuda(M[2])
            final_X = M[0].numpy()[0]
            final_Y = M[1].numpy()[0]
            loss = loss_fn(Zo, Z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            batch_loss.append(loss.detach().cpu().numpy())

        epoch_loss[e] = np.mean(batch_loss)
        if e % 5 == 0:
            print(f"Epoch={e}/{epochs},LR={lr},Averaged-Loss={epoch_loss[e]}")

        if e % checkpoint_rate == 0:
            checkpoint_path = os.path.join(mirrors_dir, dataset.name, "checkpoint", f"{e}")
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.eval().cpu(), os.path.join(checkpoint_path, "model.pt"))
            final_Z = Zmirror_prediction(model=cuda(model), dataloader=dataloader)
            final_M = (final_X, final_Y, final_Z)
            torch.save(final_M, os.path.join(checkpoint_path, "mirror.pt"))

    metadata["lr_end"] = lr

    mirror_dir = os.path.join(mirrors_dir, dataset.name)
    os.makedirs(mirror_dir, exist_ok=True)

    df = pd.DataFrame(columns=["Epoch_Loss"])
    df.Epoch_Loss = epoch_loss.tolist()
    df.to_csv(os.path.join(mirror_dir, "epoch_loss.csv"), index=False)

    # 3. generate an averaged mirror and save it to Mirrors directory
    final_Z = Zmirror_prediction(model=cuda(model), dataloader=dataloader)
    final_M = (final_X, final_Y, final_Z)
    torch.save(final_M, os.path.join(mirror_dir, "mirror.pt"))

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
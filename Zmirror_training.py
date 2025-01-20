import torch
import os
import json
import tempfile
import shutil
import pandas as pd
import numpy as np
from utils import cuda, current_time, init_weights, load_json
from ModelScripts.dataset import RaysDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from ModelScripts.models import ZMirrorLoss, ZMirrorModel

# python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

base_dir = os.path.dirname(os.path.abspath(__file__))
mirrors_dir = os.path.join(base_dir, "MirrorModels")
os.makedirs(mirrors_dir, exist_ok=True)

dir_path = os.path.join(base_dir, "Storage", "PhysicalData")

inputs = load_json("cfg/Zmirror_training_cfg.json")
dataset_path = os.path.join(dir_path, inputs['train_dir'])
batch_size = inputs['batch_size']
epochs = inputs['epochs']
lr = inputs['lr']
checkpoint_rate = inputs['checkpoint_rate']


if __name__ == "__main__":
    dt_object = current_time()
    date = dt_object.strftime('%Y%m%d')
    time = dt_object.strftime('%H%M')
    print(f"Mirror Flow Start Time = {dt_object}")

    mirror_dir = os.path.join(mirrors_dir, date + "_" + time)
    os.makedirs(mirror_dir, exist_ok=True)

    dataset = RaysDataset(root_path=dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    metadata = {
        "training_paths": dataset.mirror_dirs,
        "batch_size": batch_size,
        "lr_start": lr,
        "epochs": epochs
    }

    if torch.cuda.is_available():
        print("Training a model with CUDA GPU")
    else:
        print("Training a model with CPU")

    loss_fn = cuda(ZMirrorLoss())
    epoch_stdev = np.zeros(epochs)
    epoch_loss = np.zeros(epochs)
    model = cuda(ZMirrorModel(
        in_features=inputs['in_features'],
        in_output_features=inputs['in_output_features']
    ))
    model.apply(init_weights)
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    tmp_dir = tempfile.mkdtemp()
    models_map = dict()
    for e in range(epochs):
        batch_stdev = list()
        batch_loss = list()
        for (_, M, data), out in iter(dataloader):
            R = cuda(torch.flatten(data, start_dim=1).float())
            O = cuda(torch.flatten(out, start_dim=1).float())
            Zo = cuda(model(R=R, O=O))[0, ...]
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
        if e % 5 == 0 or e == (epochs - 1):
            temp_file = os.path.join(tmp_dir, f"{e}.pt")
            torch.save(model, temp_file)
            models_map[e] = (temp_file, epoch_loss[e])
            print(f"Epoch={e}/{epochs},{lr=},{epoch_loss[e]=},{epoch_stdev[e]=}")

    min_epoch_index = min(models_map, key=lambda k: models_map[k][-1])
    print(f"Found @ Epoch {min_epoch_index}")
    model_path, _ = models_map[min_epoch_index]
    model = torch.load(model_path)
    shutil.rmtree(tmp_dir)

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
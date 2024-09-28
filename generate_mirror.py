import pandas as pd
import torch
import os
import json
from dataset import RaysDataset
from torch.utils.data import DataLoader
from models import ZMirrorLoss
from utils import current_time


metadata = dict()

base_dir = os.path.dirname(os.path.abspath(__file__))
mirror_dir = os.path.join(base_dir, "MirrorModels")
mirror_outdir = os.path.join(mirror_dir, "GeneratedMirrors")
physical_mirrors = os.path.join(base_dir, "PhysicalMirrors")
physical_data = os.path.join(base_dir, "Storage", "PhysicalData")
mode = "valid"
rays_dataset = os.path.join(physical_data, "RaysMatrix", mode)
os.makedirs(mirror_outdir, exist_ok=True)

# pre-trained model
mirror_model_path = os.path.join(mirror_dir, "2024_09_28_21_48_21", "model.pt")
dataset = RaysDataset(root_path=rays_dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
# mirror_dir_path = os.path.join(mirror_outdir, dataset.name)
dt_object = current_time()
date = dt_object.strftime('%Y_%m_%d')
time = dt_object.strftime('%H_%M_%S')
mirror_dir_path = os.path.join(mirror_outdir, date + "_" + time)
os.makedirs(mirror_dir_path, exist_ok=True)

model = torch.load(mirror_model_path).eval()
df = pd.DataFrame(columns=['M', 'idx', 'P'])

zloss = ZMirrorLoss()
for name, M, data, i in iter(dataloader):
    name = name[0]
    i = int(i.numpy()[0])
    final_X, final_Y, Mz = M
    final_X = final_X.numpy()[0, ...]
    final_Y = final_Y.numpy()[0, ...]
    Mz = Mz.float()[0, ...]
    final_M = model(torch.flatten(data, start_dim=1).float()).detach()[0, ...]
    final_M = final_M.clamp(min=0)
    perc_loss = ((Mz - final_M).abs() / Mz).mean().item()
    loss = zloss(Mz, final_M).item()
    print(f"{name}, {i=}, {loss=}, {perc_loss=}")
    df.loc[len(df)] = [name, i, perc_loss]
    mirror_dir_path_ = os.path.join(mirror_dir_path, name)
    os.makedirs(mirror_dir_path_, exist_ok=True)
    torch.save((final_X, final_Y, final_M.numpy()), os.path.join(mirror_dir_path_, "mirror.pt"))

    metadata["mirror_model_path"] = mirror_dir_path_
    metadata["loss"] = float(loss)
    metadata["perc_loss"] = perc_loss

    with open(os.path.join(mirror_dir_path_, "metadata.json"), "w+") as f:
        json.dump(metadata, f, indent=4)

    print(f"{mirror_dir_path_=}")

df = df.groupby(by='M').agg('mean').reset_index().sort_values(by='P').drop(columns=["idx"])
print(df)
df.to_csv(os.path.join(mirror_dir_path, f"{mode}_errors.csv"), index=False)
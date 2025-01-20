import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from ModelScripts.dataset import BeamsDataset
from utils import (generate_rays_dataset,
                   load_json,
                   current_time)


base_dir = os.path.dirname(os.path.abspath(__file__))
mirror_dir = os.path.join(base_dir, "MirrorModels")
physical_mirrors_dir = os.path.join(base_dir, "PhysicalMirrors")
inputs = load_json("cfg/model_flow_exp_cfg.json")
data_name = inputs["rays_data_results_mirror_name"]
base_physical_mirror = inputs['base_physical_mirror']
model_id = inputs['model']
mirror_model_path = os.path.join(mirror_dir, model_id, "model.pt")
dt_object = current_time()
current_clock = f"{dt_object.strftime('%Y%m%d')}_{dt_object.strftime('%H%M')}"
new_phy_mirror_path = os.path.join(physical_mirrors_dir, f"{model_id}_{current_clock}")

R = pd.read_csv(inputs["rays_data_results_path"])
Ri = R.filter(regex="Ri_", axis=1).values.tolist()
Ro = R.filter(regex="Ro_", axis=1).values.tolist()
physical_list_in, amp_out = generate_rays_dataset(
    Ri=Ri,
    Ro=Ro,
    ** inputs["rays_input_boundaries"]
)

X_mirror, Y_mirror, _ = torch.load(os.path.join(physical_mirrors_dir, base_physical_mirror))
dataset = BeamsDataset(
    beams_in=[physical_list_in[0]],
    beams_out=[amp_out[0]]
)

dataloader = DataLoader(
    dataset,
    batch_size=30,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

model = torch.load(mirror_model_path).eval()
predicted_list = []
for data, out in iter(dataloader):
    predicted_mirror = model(
        R=torch.flatten(data, start_dim=1).float(),
        O=torch.flatten(out, start_dim=1).float()
    ).detach()[0, ...]
    predicted_list.append(predicted_mirror.numpy())

predicted_mean = torch.from_numpy(np.mean(predicted_list, axis=0)).clamp(min=0)

mirror_path = f"{new_phy_mirror_path}_ID_{data_name}.pt"
torch.save((X_mirror, Y_mirror, predicted_mean), mirror_path)
print(f"{mirror_path=}")

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from ModelScripts.dataset import BeamsDataset
from utils import (generate_rays_dataset,
                   load_json,
                   load_rays,
                   current_time)


base_dir = os.path.dirname(os.path.abspath(__file__))
mirror_dir = os.path.join(base_dir, "MirrorModels")
physical_mirrors_dir = os.path.join(base_dir, "PhysicalMirrors")
inputs = load_json("cfg/model_flow_imp_cfg.json")
base_physical_mirror = inputs['base_physical_mirror']
gaussian_settings = inputs['gaussian_settings']
model_id = inputs['model']
mirror_model_path = os.path.join(mirror_dir, model_id, "model.pt")
dt_object = current_time()
current_clock = f"{dt_object.strftime('%Y%m%d')}_{dt_object.strftime('%H%M')}"
new_phy_mirror_path = os.path.join(physical_mirrors_dir, f"{model_id}_{current_clock}")

print(gaussian_settings)


def generate_gaussian_amplitudes(height: int, width: int, means: list, sigmas: list) -> tuple:
    amp_out = list()
    identifiers = list()
    for xm, ym in means:
        for s in sigmas:
            x = np.arange(width)
            y = np.arange(height)
            x, y = np.meshgrid(x, y)

            amp = np.exp(-((x - xm) ** 2 + (y - ym) ** 2) / (2 * s ** 2))
            amp = amp[:, :, None]
            amp_out.append(amp)
            identifiers.append(f"{xm}_{ym}_{s}")

    return amp_out, identifiers


Ri = load_rays(path=inputs["rays_data_path"])
x_dim = inputs["rays_input_boundaries"]["x_dim"]
y_dim = inputs["rays_input_boundaries"]["y_dim"]
xi_min = inputs["rays_input_boundaries"]["xi_min"]
xi_max = inputs["rays_input_boundaries"]["xi_max"]
yi_min = inputs["rays_input_boundaries"]["yi_min"]
yi_max = inputs["rays_input_boundaries"]["yi_max"]
physical_list_in, _ = generate_rays_dataset(
    Ri=Ri,
    x_dim=x_dim,
    y_dim=y_dim,
    xi_min=xi_min,
    xi_max=xi_max,
    yi_min=yi_min,
    yi_max=yi_max,
)

amp_out, identifiers = generate_gaussian_amplitudes(**gaussian_settings)

X_mirror, Y_mirror, _ = torch.load(os.path.join(physical_mirrors_dir, base_physical_mirror))
for t, amp in enumerate(amp_out):
    idn = identifiers[t]
    tmp_amp = list()
    for i in range(len(physical_list_in)):
        tmp_amp.append(np.copy(amp))

    dataset = BeamsDataset(
        beams_in=physical_list_in,
        beams_out=tmp_amp
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = torch.load(mirror_model_path).eval()
    predicted_list = list()
    for data, out in iter(dataloader):
        predicted_mirror = model(
            R=torch.flatten(data, start_dim=1).float(),
            O=torch.flatten(out, start_dim=1).float()
        ).detach()[0, ...].clamp(min=0)
        predicted_list.append(predicted_mirror.numpy())

    predicted_mean = torch.from_numpy(np.mean(predicted_list, axis=0))

    torch.save((X_mirror, Y_mirror, predicted_mean), f"{new_phy_mirror_path}_ID_{idn}.pt")

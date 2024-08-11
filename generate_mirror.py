import torch
import os
import json
from utils import cuda
from dataset import PhysicalDataset
from torch.utils.data import DataLoader
from models import Zmirror_prediction

metadata = dict()

base_dir = os.path.dirname(os.path.abspath(__file__))
mirror_dir = os.path.join(base_dir, "MirrorModels")
os.makedirs(mirror_dir, exist_ok=True)

mirror_path = os.path.join(mirror_dir, "f96ae25e7f", "model.pt")
# mirror_path = os.path.join(base_dir, "PhysicalMirrors", "<MirrorModel filename>")

validation_mirrors = [
    os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL38.csv"),
    os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL48.csv"),
    os.path.join(base_dir, "Storage", "PhysicalData", "customized_rays_full_parabolicPFL0.5RFL68.csv")
]
dataset = PhysicalDataset(paths=validation_mirrors)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, pin_memory=True)
mirror_dir_path = os.path.join(mirror_dir, dataset.name)
os.makedirs(mirror_dir_path, exist_ok=True)

model = torch.load(mirror_path)
final_Z = Zmirror_prediction(model=cuda(model), dataloader=dataloader)
(_, _, _, _, M) = next(iter(dataloader))
final_X = M[0].numpy()[0]
final_Y = M[1].numpy()[0]
final_M = (final_X, final_Y, final_Z)
torch.save(final_M, os.path.join(mirror_dir_path, "mirror.pt"))

metadata["validation_mirrors"] = validation_mirrors
metadata["mirror_path"] = mirror_path

with open(os.path.join(mirror_dir_path, "metadata.json"), "w+") as f:
    json.dump(metadata, f, indent=4)

print(f"{mirror_dir_path=}")
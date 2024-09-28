import os
import torch
import matplotlib.pyplot as plt
from PhysicalScripts.helper import spline_mirror
from utils import current_time


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_mirror_dir_path = os.path.join(base_dir, "MirrorModels", "GeneratedMirrors")
mirror_dir_path = os.path.join(base_dir, "PhysicalMirrors")
vis_dir_path = os.path.join(base_dir, "Storage", "VisData")
os.makedirs(vis_dir_path, exist_ok=True)
show = True

model_mirror_name = "2024_09_28_21_55_15"

dt_object = current_time()
print(f"Mirror Flow Start Time = {dt_object}")

names = [
    ("parabolicPFL0.5RFL40", "super_comp_3gauss_thresh01_parabolicPFL0.5RFL40"),
    ("parabolicPFL0.5RFL50", "super_comp_3gauss_thresh01_parabolicPFL0.5RFL50"),
    ("parabolicPFL0.5RFL60", "super_comp_3gauss_thresh01_parabolicPFL0.5RFL60")
]

for phy_mirror_name, data_name in names:
    valid_mirror = os.path.join(model_mirror_dir_path, model_mirror_name, data_name, "mirror.pt")
    phy_mirror = os.path.join(mirror_dir_path, f"{phy_mirror_name}.pt")
    mirror_names = [
        (valid_mirror, f"{model_mirror_name}-valid", "red"),
        (phy_mirror, phy_mirror_name, "blue"),
    ]

    output_path = os.path.join(vis_dir_path, f"{model_mirror_name}_comparison.png")
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.set_xlim(0, 100)
    ax1.set_ylim(-24, 24)
    ax1.set_zlim(0, 70)
    ax1.view_init(10, -85)

    ax1.set_title("Mirrors Comparison")
    for mirror_path, label, color in mirror_names:
        X, Y, Z = torch.load(mirror_path)
        X, Y, Z = spline_mirror(X, Y, Z)
        ax1.plot_surface(X, Y, Z, color=color, alpha=0.5, label=label)

    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig(output_path)

print(f"Results directory: {output_path}")
print("Done!")
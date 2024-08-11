import os
import torch
import matplotlib.pyplot as plt
from PhysicalScripts.helper import spline_mirror


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_mirror_dir_path = os.path.join(base_dir, "MirrorModels")
mirror_dir_path = os.path.join(base_dir, "PhysicalMirrors")
vis_dir_path = os.path.join(base_dir, "Storage", "VisData")
os.makedirs(vis_dir_path, exist_ok=True)
show = True

model_valid_mirror_name = "9fa846e6e6"
model_train_mirror_name = "f96ae25e7f"
mirror_names = [
    (os.path.join(mirror_dir_path, "parabolicPFL0.5RFL38.pt"), "parabolicPFL0.5RFL38", "yellow"),
    (os.path.join(mirror_dir_path, "parabolicPFL0.5RFL48.pt"), "parabolicPFL0.5RFL48", "yellow"),
    (os.path.join(mirror_dir_path, "parabolicPFL0.5RFL68.pt"), "parabolicPFL0.5RFL68", "yellow"),
    (os.path.join(model_mirror_dir_path, model_valid_mirror_name, "mirror.pt"), f"{model_valid_mirror_name}-valid", "red"),
    (os.path.join(model_mirror_dir_path, model_train_mirror_name, "mirror.pt"), f"{model_train_mirror_name}-train", "blue"),
]

output_path = os.path.join(vis_dir_path, f"{model_valid_mirror_name}_comparison.png")
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
import os
import torch
import matplotlib.pyplot as plt
from PhysicalScripts.helper import spline_mirror


mirror_name = "parabolicPFL0.5RFL68"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mirror_dir_path = os.path.join(base_dir, "PhysicalMirrors")
vis_dir_path = os.path.join(base_dir, "Storage", "VisData")
os.makedirs(vis_dir_path, exist_ok=True)
show = True

output_path = os.path.join(vis_dir_path, f"{mirror_name}.png")
X, Y, Z = torch.load(os.path.join(mirror_dir_path, f"{mirror_name}.pt"))
X, Y, Z = spline_mirror(X, Y, Z)

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')

ax1.plot_surface(X, Y, Z, color="blue", alpha=0.5, label="b")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1.set_xlim(0, 100)
ax1.set_ylim(-24, 24)
ax1.set_zlim(0, 70)
ax1.view_init(10, -85)

ax1.set_title(mirror_name)
if show:
    plt.show()
else:
    plt.savefig(output_path)
print(f"Results directory: {output_path}")
print("Done!")
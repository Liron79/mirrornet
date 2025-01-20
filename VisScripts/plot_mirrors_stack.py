import os
import torch
import matplotlib.pyplot as plt
from PhysicalScripts.helper import spline_mirror
from utils import current_time


save = False
mirror_tuples = [
    # (label, unique color, file name)
]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mirror_dir_path = os.path.join(base_dir, "PhysicalMirrors")
vis_dir_path = os.path.join(base_dir, "Storage", "VisData", "mirrors")
os.makedirs(vis_dir_path, exist_ok=True)

dt_object = current_time()
print(f"Mirror Flow Start Time = {dt_object}")

output_path = os.path.join(vis_dir_path, f"{mirror_tuples[-1][-1]}.png")
fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1.set_xlim(0, 100)
ax1.set_ylim(-24, 24)
ax1.set_zlim(0, 70)
ax1.view_init(10, -85)
ax1.set_title("Mirrors Shape Validation")

for label, color, data_name in mirror_tuples:
    X, Y, Z = torch.load(os.path.join(mirror_dir_path, f"{data_name}.pt"))
    X, Y, Z = spline_mirror(X, Y, Z)
    ax1.plot_surface(X, Y, Z, color=color, alpha=0.5, label=label)

plt.legend()
if save:
    plt.savefig(output_path)
plt.show()


print(f"Results directory: {output_path}")
print("Done!")
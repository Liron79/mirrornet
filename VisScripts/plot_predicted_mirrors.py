import os
import torch
import os
import matplotlib.pyplot as plt
from PhysicalScripts.helper import spline_mirror

mirror_key = "1920308aa1"
mirror_name = "mirror.pt"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mirror_dir_path = os.path.join(base_dir, "Mirrors")
vis_dir_path = os.path.join(base_dir, "VisData")
os.makedirs(vis_dir_path, exist_ok=True)
show = True
output_path = os.path.join(vis_dir_path, f"{mirror_key}.png")
X, Y, Z = torch.load(os.path.join(mirror_dir_path, mirror_key, mirror_name))
X, Y, Z = spline_mirror(X, Y, Z)

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')

ax1.plot_surface(X, Y, Z, color="green", alpha=0.5, label="b")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(mirror_key)
if show:
    plt.show()

plt.savefig(output_path)
print(f"Results directory: {output_path}")
print("Done!")
import os
import torch
import matplotlib.pyplot as plt
from PhysicalScripts.helper import spline_mirror

M1 = "<mirror data name>"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mirror_dir_path = os.path.join(base_dir, "PhysicalMirrors")
vis_dir_path = os.path.join(base_dir, "Storage", "VisData")
os.makedirs(vis_dir_path, exist_ok=True)
mirror_path = os.path.join(mirror_dir_path, f"{M1}.pt")

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1.set_xlim(0, 80)
ax1.set_ylim(-24, 24)
ax1.set_zlim(0, 70)
ax1.view_init(10, -85)

X, Y, Z = torch.load(mirror_path)
X, Y, Z = spline_mirror(X, Y, Z)
# ax1.plot_surface(X, Y, Z, color="green", alpha=0.5, label=f"{M1}")
ax1.plot_surface(X, Y, Z, color="green", alpha=0.5)


plt.legend()
plt.show()

print("Done!")
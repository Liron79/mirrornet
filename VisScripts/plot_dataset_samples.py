import os
import numpy as np
from matplotlib import pyplot as plt


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rays_dir_path = os.path.join(base_dir, "Storage", "PhysicalData", "RaysMatrix", "train")

path = os.path.join(rays_dir_path, "super_comp_3gauss_thresh001_parabolicPFL0.5RFL46")
data_id = 11
Ri_path = os.path.join(path, "inputs", f"{data_id}.npy")
Ro_path = os.path.join(path, "outputs", f"{data_id}.npy")

Ri = np.load(Ri_path) # NxNx2
Ro = np.load(Ro_path) # MxMx2

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2)
ax1[0].imshow(Ri[:, :, 0])
ax1[0].set_title("Kxi")
ax1[0].axis('off')
ax1[1].imshow(Ri[:, :, 1])
ax1[1].set_title("Kyi")
ax1[1].axis('off')
ax2[0].imshow(Ro[:, :, 0])
ax2[0].set_title("Kyo")
ax2[0].axis('off')
ax2[1].imshow(Ro[:, :, 1])
ax2[1].set_title("Kzo")
ax2[1].axis('off')

plt.tight_layout()
plt.show()
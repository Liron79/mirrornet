import os
import numpy as np
from matplotlib import pyplot as plt


target_file = "<mirror name>"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rays_dir_path = os.path.join(base_dir, "Storage", "PhysicalData", "train_M4")

path = os.path.join(rays_dir_path, target_file)
data_id = 8
Ri_path = os.path.join(path, "inputs", f"{data_id}.npy")
Ro_path = os.path.join(path, "outputs", f"{data_id}.npy")

Ri = np.load(Ri_path)
Ro = np.load(Ro_path)
print(f"{Ri.shape=}", f"{Ro.shape=}")
print(Ri.max(), Ri.min())
print(Ro.max(), Ro.min())

# plt.hist(Ro.ravel(), bins=50, edgecolor='black')
# plt.show()
# counts, bin_edges = np.histogram(Ri.ravel(), bins=50)
# plt.bar(bin_edges[:-1], counts, width=(bin_edges[1] - bin_edges[0]), edgecolor='black')
# plt.show()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.imshow(Ri[:, :, 0])
ax1.set_title("Amp Input")
ax1.axis('off')
ax2.imshow(Ro[:, :, 0])
ax2.set_title("Amp Output")
ax2.axis('off')

plt.tight_layout()
plt.show()
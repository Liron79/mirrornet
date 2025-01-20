import os.path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import load_json


inputs = load_json("cfg/eval_model_cfg.json")
data_ref = inputs["data_ref"]
gt_ref = inputs["gt_ref"]
gt_dir = inputs["gt_dir"]
data_name = inputs["data_name"]
data_dir = inputs["data_dir"]
y_dim = inputs["y_dim"]
z_dim = inputs["z_dim"]
base_dir = os.path.dirname(os.path.abspath(__file__))
physical_gt_dir = os.path.join(base_dir, "Storage", "PhysicalData", gt_dir)
physical_data_dir = os.path.join(base_dir, "Storage", "PhysicalData", data_dir)
rays_out_dir = os.path.join(base_dir, "Storage", "RaysOut")
path = os.path.join(physical_data_dir, data_name, f"{data_name}.csv")
ref_path = os.path.join(physical_gt_dir, gt_ref, f"{gt_ref}.csv")
gauss_path = os.path.join(rays_out_dir, data_ref)

ref_matrix = np.zeros((y_dim, z_dim))
matrix = np.zeros((y_dim, z_dim))
gt_gauss = np.load(gauss_path)[:, :, 0]
df = pd.read_csv(path)
ref_df = pd.read_csv(ref_path)

ref_df.Ro_y = ref_df.Ro_y.astype(np.int64)
ref_df.Ro_y = ref_df.Ro_y + y_dim // 2
ref_df.Ro_z = ref_df.Ro_z.astype(np.int64)

for (y, z), grp in ref_df.groupby(by=["Ro_y", "Ro_z"]):
    if y >= y_dim or z >= z_dim:
        continue
    ref_matrix[y, z] = grp.Ro_amp.sum()

df.Ro_y = df.Ro_y.astype(np.int64)
df.Ro_y = df.Ro_y + y_dim // 2
df.Ro_z = df.Ro_z.astype(np.int64)

for (y, z), grp in df.groupby(by=["Ro_y", "Ro_z"]):
    if y >= y_dim or z >= z_dim or y < 0 or z < 0:
        continue
    matrix[y, z] = grp.Ro_amp.sum() / df.Ro_amp.sum()

y_labels = [str(x) for x in range(-y_dim // 2 + 1, y_dim // 2 + 2, 2)]
z_labels = [str(x) for x in range(0, z_dim + 1, 2)]

print(matrix)

plt.subplot(2, 1, 1)
plt.yticks(range(0, y_dim + 1, 2), y_labels)
plt.xticks(range(0, z_dim + 1, 2), z_labels, rotation=90)
plt.imshow(ref_matrix)
plt.title("Reference Outputs")
plt.colorbar()
plt.subplot(2, 1, 2)
plt.yticks(range(0, y_dim + 1, 2), y_labels)
plt.xticks(range(0, z_dim + 1, 2), z_labels, rotation=90)
plt.imshow(matrix)
plt.title("Predicted Outputs")
plt.colorbar()
# plt.subplot(3, 1, 3)
# plt.title("Ground Truth")
# plt.imshow(gt_gauss)
# plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(physical_data_dir, f"{data_name}.png"))
plt.show()
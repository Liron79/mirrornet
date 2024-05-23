import os
import json
import pandas as pd
import numpy as np
from VisScripts.vis_utils import plot_metrics
from dataset import cell_sampling

mirror_name = "e8ce93b732"
validated_dir = "ed196b09b9"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M1_dir = os.path.join(base_dir, "Mirrors", mirror_name)
dir_path = os.path.join(base_dir, "PhysicalData", mirror_name)
vis_path = os.path.join(base_dir, "VisData", validated_dir)
os.makedirs(vis_path, exist_ok=True)
data_path = os.path.join(dir_path, validated_dir, "full_data.csv")

n = 11
A_MSE = np.zeros((n, n))
A_AMP = np.zeros((n, n))
A_AMP_PRED = np.zeros((n, n))
A_NUM_LOSSES = np.zeros((n, n))
A_PERC_LOSSES = np.zeros((n, n))
show = True
# group by cell
X_R = range(30, 52, 2)
Y_R = range(-10, 12, 2)
with open(os.path.join(M1_dir, "metadata.json")) as f:
    cell_resolution = json.load(f)["cell_resolution"]

df = pd.read_csv(data_path)
df_mirror = df[~df.M.str.contains(mirror_name)]
df_mirror_pred = df[df.M.str.contains(mirror_name)]

X_range = {x:i for i, x in enumerate(X_R)}
Y_range = {y:i for i, y in enumerate(Y_R)}

data_df = pd.DataFrame(index=["mse", "amp", "amp_pred", "num_losses", "perc_losses"])
for (x, y), grp in df_mirror.groupby(by=["Ri_x", "Ri_y"]):
    # grp = cell_sampling(data=grp, cell_resolution=cell_resolution)
    grp_pred = df_mirror_pred[(df_mirror_pred.Ri_x == x)&(df_mirror_pred.Ri_y == y)]
    # grp_pred = cell_sampling(data=grp_pred, cell_resolution=cell_resolution)
    if grp_pred.empty or len(grp_pred) != len(grp):
        continue
    x_new = X_range[x]
    y_new = Y_range[y]
    MSE_xy = (grp.Ro_x.values - grp_pred.Ro_x.values)**2
    MSE_xy += (grp.Ro_y.values - grp_pred.Ro_y.values)**2
    MSE_xy += (grp.Ro_z.values - grp_pred.Ro_z.values)**2
    A_MSE[x_new, y_new] = np.sqrt(MSE_xy).mean()
    A_AMP[x_new, y_new] = grp.Ro_amp.sum()
    A_AMP_PRED[x_new, y_new] = grp_pred.Ro_amp.sum()
    uniq_xy = len(set(grp.Ro_ray_index.values))
    pred_uniq_xy = len(set(grp_pred.Ro_ray_index.values))
    num_losses = uniq_xy - pred_uniq_xy
    A_NUM_LOSSES[x_new, y_new] = num_losses
    percentage_losses = 1.0 - pred_uniq_xy / uniq_xy
    A_PERC_LOSSES[x_new, y_new] = percentage_losses
    # generate Kx, Ky graphs: real vs prediction
    data_df[f"{x}_{y}"] = [A_MSE[x_new, y_new], A_AMP[x_new, y_new],
                           A_AMP_PRED[x_new, y_new], A_NUM_LOSSES[x_new, y_new],
                           A_PERC_LOSSES[x_new, y_new]]

data_df.to_csv(os.path.join(vis_path, "data_cell.csv"), index=True)

X_R, Y_R = np.meshgrid(X_R, Y_R)
plot_metrics(X=X_R, Y=Y_R, Z=A_AMP, path=os.path.join(vis_path, "amp_cell.png"),
             x_label="", y_label="", z_label="", z_factor=1.0, title="", show=show)
plot_metrics(X=X_R, Y=Y_R, Z=A_AMP_PRED, path=os.path.join(vis_path, "amp_cell_pred.png"),
             x_label="", y_label="", z_label="", z_factor=1.0, title="", show=show)

print(f"Results directory: {vis_path}")
print("Done!")
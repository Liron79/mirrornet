import os
import pandas as pd
import numpy as np
from VisScripts.vis_utils import plot_metrics

mirror_name = "5c3d14b758"
data_name = f"07a1c9f03c"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_path = os.path.join(base_dir, "PhysicalData", mirror_name)
vis_path = os.path.join(base_dir, "VisData", data_name)
os.makedirs(vis_path, exist_ok=True)
data_path = os.path.join(dir_path, data_name, "full_data.csv")

n = 11
A_MSE = np.zeros((n, n))
A_AMP = np.zeros((n, n))
A_AMP_PRED = np.zeros((n, n))
A_NUM_LOSSES = np.zeros((n, n))
A_PERC_LOSSES = np.zeros((n, n))
show = False
# group by cell
X_R = range(30, 52, 2)
Y_R = range(-10, 12, 2)

df = pd.read_csv(data_path)
df_mirror = df[~df.M.str.contains(mirror_name)]
df_mirror_pred = df[df.M.str.contains(mirror_name)]

X_range = {x:i for i, x in enumerate(X_R)}
Y_range = {y:i for i, y in enumerate(Y_R)}

data_df = pd.DataFrame(index=["MSE", "AMP", "AMP_PRED", "NUM_LOSSES", "PERC_LOSSES"])
for (x, y), grp in df_mirror.groupby(by=["Ri_x", "Ri_y"]):
    grp_pred = df_mirror_pred[(df_mirror_pred.Ri_x == x)&(df_mirror_pred.Ri_y == y)]
    if grp_pred.empty:
        continue
    x_new = X_range[x]
    y_new = Y_range[y]
    MSE_xy = (grp.Ro_x.values - grp_pred.Ro_x.values)**2
    MSE_xy += (grp.Ro_y.values - grp_pred.Ro_y.values)**2
    MSE_xy += (grp.Ro_z.values - grp_pred.Ro_z.values)**2
    A_MSE[x_new, y_new] = MSE_xy.mean()
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

data_df.to_csv(os.path.join(vis_path, "DATA_CELL.csv"), index=True)
X_R, Y_R = np.meshgrid(X_R, Y_R)
plot_metrics(X=X_R, Y=Y_R, Z=A_MSE, path=os.path.join(vis_path, "MSE_CELL.png"),
             x_label="", y_label="", z_label="", title="", show=show)
plot_metrics(X=X_R, Y=Y_R, Z=A_AMP, path=os.path.join(vis_path, "AMP_CELL.png"),
             x_label="", y_label="", z_label="", title="", show=show)
plot_metrics(X=X_R, Y=Y_R, Z=A_AMP_PRED, path=os.path.join(vis_path, "AMP_CELL_PRED.png"),
             x_label="", y_label="", z_label="", title="", show=show)
plot_metrics(X=X_R, Y=Y_R, Z=A_NUM_LOSSES, path=os.path.join(vis_path, "NUM_LOSSES_CELL.png"),
             x_label="", y_label="", z_label="", title="", show=show)
plot_metrics(X=X_R, Y=Y_R, Z=A_PERC_LOSSES, path=os.path.join(vis_path, "PERC_LOSSES_CELL.png"),
             x_label="", y_label="", z_label="", title="", show=show)

print(f"Results directory: {vis_path}")
print("Done!")
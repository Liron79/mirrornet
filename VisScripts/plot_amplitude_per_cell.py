import os
import pandas as pd
import numpy as np
from VisScripts.vis_utils import plot_metrics


plot_title = "Amplitude per Cell"
show = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
phy_path = os.path.join(base_dir, "PhysicalData")
vis_path = os.path.join(base_dir, "VisData")
data_path = os.path.join(phy_path, "pulse_1x6x6_parabolic.csv")
df = pd.read_csv(data_path)
BY = ["Ro_y", "Ro_z"]
# pad_factor = 1
# Y_max = int(df[BY[0]].max() + 1)
# Y_min = int(df[BY[0]].min() - 1)
# Y_range = list(np.arange(Y_min - pad_factor, Y_max + pad_factor, 1))
# Z_max = int(df[BY[1]].max() + 1)
# Z_min = int(df[BY[1]].min() - 1)
# Z_range = list(np.arange(Z_min - pad_factor, Z_max + pad_factor, 1))
Y_range = range(0, 1, 1)
Z_range = range(15, 26, 1)
Z_AMP = np.zeros((len(Y_range), len(Z_range)))
map_y = {y: i for i, y in enumerate(Y_range)}
map_z = {z: j for j, z in enumerate(Z_range)}

n = 11
A_AMP = np.zeros((n, n))

for (y, z), grp in df.groupby(by=BY):
    y = int(y)
    z = int(z)
    if grp.empty:
        continue
    if y == 0 and 15 < z < 25:
        y_new = map_y[y]
        z_new = map_z[z]
        A_AMP[y_new, z_new] = grp.Ro_amp.sum()

area = "[mVs/m]"
decimals = 2
ZAXIS_label = "|Ex| {}\u00b2".format(area, decimals)
map_y, map_z = np.meshgrid(map_y, map_z)
plot_metrics(X=map_y, Y=map_z, Z=A_AMP, path=os.path.join(vis_path, "amp_per_cell.png"),
             x_label="Y [mm]", y_label="Z [mm]", z_label=ZAXIS_label, z_factor=1e6, title=plot_title, show=show)

print(f"Results directory: {vis_path}")
print("Done!")
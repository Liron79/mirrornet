import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pulse_title = "<RaysIn filename>"
pulse_name = "<Title>"
show = True

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pulse_dir_path = os.path.join(base_dir, "Storage", "RaysIn")
vis_dir_path = os.path.join(base_dir, "Storage", "VisData")
os.makedirs(vis_dir_path, exist_ok=True)

pulse_path = os.path.join(pulse_dir_path, f"{pulse_name}.csv")
output_path = os.path.join(vis_dir_path, f"rays_in_{pulse_name}.png")

XAXIS_label = "X [mm]"
YAXIS_label = "Y [mm]"
area = "[mVs/m]"
decimals = 2
ZAXIS_label = "|Ex| {}\u00b2".format(area, decimals)
BY = ["x", "y"]
AMP_name = "amp"

df = pd.read_csv(pulse_path)
pad_factor = 1
X_max = int(df[BY[0]].max() + 1)
X_min = int(df[BY[0]].min() - 1)
X_range = list(range(X_min - pad_factor, X_max + pad_factor, 1))
Y_max = int(df[BY[1]].max() + 1)
Y_min = int(df[BY[1]].min() - 1)
Y_range = list(range(Y_min - pad_factor, Y_max + pad_factor, 1))
Z_AMP = np.zeros((len(X_range), len(Y_range)))
map_x = {x: i for i, x in enumerate(X_range)}
map_y = {y: j for j, y in enumerate(Y_range)}
for name, grp in df.groupby(by=BY):
    x, y = name
    Z_AMP[map_x[int(x)], map_y[int(y)]] = grp[AMP_name].mean()

X, Y = np.meshgrid(X_range, Y_range)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z_AMP.T * 1e6, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel(XAXIS_label, fontsize=10, weight='semibold')
ax.set_ylabel(YAXIS_label, fontsize=10, weight='semibold')
ax.set_zlabel(ZAXIS_label, fontsize=10, weight='semibold')
ax.set_title(pulse_title)
if show:
    plt.show()
else:
    plt.savefig(output_path)
print(f"Results directory: {output_path}")
print("Done!")
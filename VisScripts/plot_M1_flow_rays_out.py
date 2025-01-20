import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pulse_name = "<physical data name>"
pulse_title = "<physical data name>"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pulse_dir_path = os.path.join(base_dir, "Storage", "PhysicalData", "archive", pulse_name)
vis_dir_path = os.path.join(base_dir, "Storage", "VisData")
os.makedirs(vis_dir_path, exist_ok=True)

pulse_path = os.path.join(pulse_dir_path, f"{pulse_name}.csv")
output_path = os.path.join(vis_dir_path, f"rays_out_{pulse_name}.png")
show = True

XAXIS_label = "Y axis"
YAXIS_label = "Z axis"
# area = "[mVs/m]"
decimals = 2
# ZAXIS_label = "|Ex| {}\u00b2".format(area, decimals)
BY = ["Ro_y", "Ro_z"]
AMP_name = "Ro_amp"

df = pd.read_csv(pulse_path)
df = df[(abs(df.Ro_y) <= 5)&(15 <= df.Ro_z)&(df.Ro_z <= 25)]
print(f"Number of samples: {df.shape[0]}")
print(f"Number of samples after filtering: {df.shape[0]}")
X_max = round(df[BY[0]].max())
X_min = round(df[BY[0]].min())
X_range = list(np.arange(X_min, X_max + 1, 1))
Y_max = round(df[BY[1]].max())
Y_min = round(df[BY[1]].min())
Y_range = list(np.arange(Y_min, Y_max + 1, 1))
Z_AMP = np.zeros((len(X_range), len(Y_range)))
map_y = {round(x): i for i, x in enumerate(X_range)}
map_z = {round(y): j for j, y in enumerate(Y_range)}
print("map_y:", map_y)
print("map_y:", map_z)
is_inrange = lambda y, z: -1 <= y <= 1 and 19 <= z <= 21
for name, grp in df.groupby(by=BY):
    y, z = name
    y = round(y)
    z = round(z)
    if is_inrange(y, z):
        if y > abs(max(map_y.keys())) or z > abs(max(map_z.keys())):
            continue
        new_y = map_y[y]
        new_z = map_z[z]
        amp = grp[AMP_name].mean()
        Z_AMP[new_y, new_z] += amp

X, Y = np.meshgrid(X_range, Y_range)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z_AMP.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel(XAXIS_label, fontsize=10, weight='semibold')
ax.set_ylabel(YAXIS_label, fontsize=10, weight='semibold')
if show:
    plt.show()
else:
    plt.savefig(output_path)

print(f"Results directory: {output_path}")
print("Done!")
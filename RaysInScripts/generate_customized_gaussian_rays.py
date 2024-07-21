import os
import csv
import numpy as np


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raysin_dir_path = os.path.join(base_dir, "Storage", "RaysIn")
os.makedirs(raysin_dir_path, exist_ok=True)


with (open(os.path.join(raysin_dir_path, "customized_gaussian_region_rays.csv"), "w+", newline="") as f):
    columns = ["x", "y", "z", "kx", "ky", "kz", "ex", "ey", "ez", "distance", "amp", "status", "ray_index"]
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(columns)
    idx = 1
    for i in np.arange(38, 42, 0.25):
        for j in np.arange(-2, 2, 0.25):
            kx = np.random.normal(0, 1)
            ky = np.random.normal(0, 1)
            kz = -1
            Ri = [i, j, 60, kx, ky, kz, 1, 0, 0, 0, 1, 0, idx]
            idx += 1
            csv_writer.writerow(Ri)
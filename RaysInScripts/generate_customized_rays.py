import os
import csv
import numpy as np


RAND_ANGLES = False
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raysin_dir_path = os.path.join(base_dir, "Storage", "RaysIn")
os.makedirs(raysin_dir_path, exist_ok=True)


with open(os.path.join(raysin_dir_path, "customized_rays_full.csv"), "w+", newline="") as f:
    columns = ["x", "y", "z", "kx", "ky", "kz", "ex", "ey", "ez", "distance", "amp", "status", "ray_index"]
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(columns)
    idx = 1
    for i in np.arange(32, 54, 2):
        for j in np.arange(-10, 12, 2):
            Ri = [i, j, 60, 0, 0, -1, 1, 0, 0, 0, 1, 0, idx]
            idx += 1
            csv_writer.writerow(Ri)
            if RAND_ANGLES:
                for _ in range(5):
                    Ri = [i, j, 60, np.random.rand(), np.random.rand(), -1, 1, 0, 0, 0, 1, 0, idx]
                    idx += 1
                    csv_writer.writerow(Ri)
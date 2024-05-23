import csv
import numpy as np


dim_x, dim_y, dim_z = 100, 100, 100

x, y = np.meshgrid(np.linspace(-dim_x, dim_y, dim_z), np.linspace(-dim_x, dim_y, dim_z))
grid_rays = np.where(abs(x) <= dim_x, 1, 0) & np.where(abs(y) <= dim_y, 1, 0)
Nx, Ny = grid_rays.shape

idx = 1
columns = ["x", "y", "z", "kx", "ky", "kz", "ex", "ey", "ez", "distance", "amp", "status", "ray_index"]
with open("C:/Users/User/PycharmProjects/mirrornet/RaysIn/customized_centered_rays.csv", "w+", newline="") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(columns)
    for i in np.arange(38, 42, 0.25):
        for j in np.arange(-2, 2, 0.25):
            Ri = [i, j, 60, 0, 0, -1, 1, 0, 0, 0, 1, 0, idx]
            idx += 1
            csv_writer.writerow(Ri)
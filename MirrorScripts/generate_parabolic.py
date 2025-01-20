import numpy as np
import torch
import os


def generate_parabolic(RFL: int = 40, PFL: int = 0.5, random_flag: bool = False) -> list:
    bord = 0  # borders
    steps = 2

    x_step = steps
    y_step = steps

    # RFL = 40  # Reflective Focal Length
    # PFL = 0.5 * RFL  # Parent Focal Length
    RFL = RFL  # Reflective Focal Length
    PFL = PFL * RFL  # Parent Focal Length

    x_length = 48 + 2 * bord * x_step
    y_length = 48 + 2 * bord * y_step
    x_offaxis = 40
    y_offaxis = 0

    ka = 0.25 / PFL

    m = int(x_length / x_step + 1)
    n = int(y_length / y_step + 1)

    x = np.zeros(m)
    y = np.zeros(n)
    z = np.zeros(n * m)

    M = list()
    for i in range(m):
        x[i] = (-(m - 1) / 2 + (i)) * x_step + x_offaxis
    M.append(x)

    for j in range(n):
        y[j] = (-(n - 1) / 2 + (j)) * y_step + y_offaxis
    M.append(y)

    for i in range(m):
        for j in range(n):
            z[j + i * n] = ka * (np.power(x[i], 2) + np.power(y[j], 2))
            if random_flag:
                z[j + i * n] = z[j + i * n] + np.random.rand(*z[j + i * n].shape)
    M.append(z)

    return M


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mirrors_dir_path = os.path.join(base_dir, "PhysicalMirrors")
    os.makedirs(mirrors_dir_path, exist_ok=True)

    random_flag = False
    RFL_list = [43, 47]
    PFL_List = [0.5]
    random_factor = 1
    for rfl in RFL_list:
        for pfl in PFL_List:
            for i in range(1, random_factor + 1):
                M = generate_parabolic(RFL=rfl, PFL=pfl, random_flag=random_flag)
                rand_title = 'rand' if random_flag else ''
                # torch.save(M, os.path.join(mirrors_dir_path, f"parabolicPFL{pfl}RFL{rfl}_{rand_title}_{i}.pt"))
                torch.save(M, os.path.join(mirrors_dir_path, f"parabolicPFL{pfl}RFL{rfl}.pt"))
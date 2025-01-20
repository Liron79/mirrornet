import os
import torch
import numpy as np
from utils import (load_json,
                   load_rays,
                   generate_physical_data,
                   current_time,
                   generate_rays_dataset)
from PhysicalScripts import RTR_MT_M1_XY_input, RTR_MT_M2_YZ_input, RTR_M1_XY_input


base_dir = os.path.dirname(os.path.abspath(__file__))
inputs = load_json("cfg/mirror_flow_cfg.json")
physical_mirrors = os.path.join(base_dir, "PhysicalMirrors")
physical_data_dir = os.path.join(base_dir, "Storage", "PhysicalData")
M1_dir_out = inputs["M1_dir_out"]
M1_name_list = inputs["M1_name_list"]
rays_path = inputs["rays_data_path"]

os.makedirs(physical_data_dir, exist_ok=True)

if __name__ == "__main__":
    dt_object = current_time()
    print(f"Mirror Flow Start Time = {dt_object}")

    for M1_name in M1_name_list:
        print(f"Processing Mirror: {M1_name}")
        M1_path = os.path.join(physical_mirrors, f"{M1_name}.pt")
        M = torch.load(M1_path)
        print(f"Loading Mirror M1 from: {M1_path}")

        Ri = load_rays(path=rays_path)
        print(f"Loading Rays Ri (Count={len(Ri)}) from: {rays_path}")

        print("Applying physical reflection of Ri on M1...")
        Ri, Rint, Ro = RTR_MT_M1_XY_input.calcRayIntersect(Ri, M, show_plot=False)

        Ri = list(Ri)
        Rint = list(Rint)
        Ro = list(Ro)

        print("Generating Dataframe...")
        physical_data = generate_physical_data(Ri, Rint, Ro, M1_path)

        print("Generating Datasets...")
        physical_list_in, physical_list_out = generate_rays_dataset(
            Ri=Ri,
            Ro=Ro,
            **inputs["rays_input_boundaries"]
        )
        print("Summary table:")
        print(physical_data)

        filename = f"{os.path.basename(rays_path).split('.')[0]}_{M1_name}"
        ray_matrix_dir = os.path.join(physical_data_dir, M1_dir_out, filename)
        os.makedirs(ray_matrix_dir, exist_ok=True)
        physical_data.to_csv(os.path.join(ray_matrix_dir, f"{filename}.csv"), index=False)
        with open(os.path.join(ray_matrix_dir, "mirror.txt"), "w+") as f:  # mirror source path
            f.write(M1_path)
        ray_matrix_input_dir = os.path.join(ray_matrix_dir, "inputs")
        ray_matrix_output_dir = os.path.join(ray_matrix_dir, "outputs")
        os.makedirs(ray_matrix_input_dir, exist_ok=True)
        os.makedirs(ray_matrix_output_dir, exist_ok=True)
        for i, (physical_matrix_in, physical_matrix_out) in enumerate(zip(physical_list_in, physical_list_out)):
            np.save(os.path.join(ray_matrix_input_dir, f"{i}.npy"), physical_matrix_in)
            np.save(os.path.join(ray_matrix_output_dir, f"{i}.npy"), physical_matrix_out)

        dt_object_end = current_time()
        print(f"Mirror Flow End Time = {dt_object_end}")
        print(f"Mirror Flow Duration (Sec) = {(dt_object_end - dt_object).total_seconds()}")
        print(ray_matrix_dir)

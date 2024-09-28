import os
import torch
import numpy as np
from utils import (load_json, load_rays, physical_rays_drop,
                   generate_physical_data, current_time, transform,
                   generate_rays_dataset)
from PhysicalScripts import RTR_MT_M1_XY_input, RTR_MT_M2_YZ_input, RTR_M1_XY_input

# Loading inputs from config file
base_dir = os.path.dirname(os.path.abspath(__file__))
inputs = load_json("mirror_flow_cfg.json")
physical_mirrors = os.path.join(base_dir, "PhysicalMirrors")
M1_dir = inputs["M1_dir"]
M1_key = inputs["M1_key"]
M1_name = inputs["M1_name"]
M2_run = inputs["M2_run"]
M2_name = inputs["M2_name"]
rays_path = inputs["rays_data_path"]
physical_data_dir = inputs["physical_data_dir"]
rays_data_dir = os.path.join(physical_data_dir, "RaysMatrix")
os.makedirs(physical_data_dir, exist_ok=True)

if __name__ == "__main__":
    dt_object = current_time()
    print(f"Mirror Flow Start Time = {dt_object}")

    # 1. Load a mirror
    M1_path = os.path.join(M1_dir, f"{M1_name}.pt")
    M = torch.load(M1_path)
    print(f"Loading Mirror M1 from: {M1_path}")

    # 2. load rays input
    Ri = load_rays(path=rays_path)
    print(f"Loading Rays Ri (Count={len(Ri)}) from: {rays_path}")

    # 3. rays output generation
    print("Applying physical reflection of Ri on M1...")
    Ri, Rint, Ro = RTR_MT_M1_XY_input.calcRayIntersect(Ri, M, show_plot=False)
    # Rint, Ro = RTR_M1_XY_input.calcRayIntersect(Ri, M, show_plot=False) # TODO: serial function
    Ri = list(Ri)
    Rint = list(Rint)
    Ro = list(Ro)

    # 4. data output generation
    print("Generating Dataframe...")
    physical_data = generate_physical_data(Ri, Rint, Ro, M1_path)

    # 5. dataset generation for the neural network modeling
    print("Generating Datasets...")
    physical_list_in, physical_list_out = generate_rays_dataset(Ri, Ro, **inputs["rays_input_boundaries"])
    print("Summary table:")
    print(physical_data)

    # 6. save dataframe of rays
    M1_title = f"{M1_key}_{M1_name}" if M1_key is not None and len(M1_key) > 0 else f"{M1_name}"
    filename = f"{os.path.basename(rays_path).split('.')[0]}_{M1_title}"
    physical_data.to_csv(os.path.join(physical_data_dir, f"{filename}.csv"), index=False)
    # 7. save dataset items for the neural network
    ray_matrix_dir = os.path.join(rays_data_dir, filename)
    os.makedirs(ray_matrix_dir, exist_ok=True)
    with open(os.path.join(ray_matrix_dir, "mirror.txt"), "w+") as f:  # mirror source path
        f.write(M1_path)
    ray_matrix_input_dir = os.path.join(ray_matrix_dir, "inputs")
    ray_matrix_output_dir = os.path.join(ray_matrix_dir, "outputs")
    os.makedirs(ray_matrix_input_dir, exist_ok=True)
    os.makedirs(ray_matrix_output_dir, exist_ok=True)
    for i, (physical_matrix_in, physical_matrix_out) in enumerate(zip(physical_list_in, physical_list_out)):
        np.save(os.path.join(ray_matrix_input_dir, f"{i}.npy"), physical_matrix_in)
        np.save(os.path.join(ray_matrix_output_dir, f"{i}.npy"), physical_matrix_out)

    # TODO: check if second mirror is required by the Prof.
    """
    # 8. Rerun flow with the output rays onto another mirror
    if M2_run:
        M2_path = os.path.join(physical_mirrors, M2_name)
        print(f"Loading Mirror M2 from: {M2_path}")
        M2 = torch.load(M2_path)
        print(f"Transforming Rays Ro to Ri#2 (Count={len(Ro)})")
        Ri2 = transform(Ro, kind="linear")
        print("Applying physical reflection of Ri#2 on M2...")
        Rint2, Ro2 = RTR_MT_M2_YZ_input.calcRayIntersectM2(Ri2, M2, show_plot=False)
        Ro2 = Ro2.tolist()
        Rint2 = Rint2.tolist()
        print("Dropping invalid rays of Ro#2...")
        Ri2 = physical_rays_drop(Ri2, Ro2)
        print("Done!")
        physical_data = generate_physical_data(Ri2, Rint2, Ro2, M2_path)
        print("Summary table:")
        print(physical_data)
        # save dataframe output - assumption: need to be identical to the input rays
        filename = f"{filename}_{os.path.basename(M2_path).split('.')[0]}"
        physical_data.to_csv(os.path.join(physical_data_dir, f"{filename}.csv"), index=False)
    """

    dt_object_end = current_time()
    print(f"Mirror Flow End Time = {dt_object_end}")
    print(f"Mirror Flow Duration (Sec) = {(dt_object_end - dt_object).total_seconds()}")

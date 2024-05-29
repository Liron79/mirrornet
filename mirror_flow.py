import os
import torch
from utils import load_rays, physical_rays_drop, generate_physical_data, current_time, transform
from PhysicalScripts import RTR_M1_XY_input, RTR_M2_YZ_input
from PhysicalScripts import RTR_MT_M1_XY_input, RTR_MT_M2_YZ_input

base_dir = os.path.dirname(os.path.abspath(__file__))
M1_name = "parabolic"
# M1_name = "mirror"
M1_key = "d9f4223526" if M1_name == "mirror" else None
M1_dir = os.path.join(base_dir, "Mirrors")
rays_path = os.path.join(base_dir, "RaysIn", "pulse_1x6x6.csv")
physical_data_dir = os.path.join(base_dir, "PhysicalData")
os.makedirs(physical_data_dir, exist_ok=True)


if __name__ == "__main__":
    dt_object = current_time()
    print(f"Mirror Flow Start Time = {dt_object}")

    # 1. Load a mirror
    if M1_key is None or len(M1_key) == 0:
        M1_path = os.path.join(M1_dir, f"{M1_name}.pt")
    else:
        M1_path = os.path.join(M1_dir, M1_key, f"{M1_name}.pt")
    M = torch.load(M1_path)
    print(f"Loading Mirror M1 from: {M1_path}")

    # 2. load rays in
    Ri = load_rays(path=rays_path)
    print(f"Loading Rays Ri (Count={len(Ri)}) from: {rays_path}")

    # 3. reflection - generate rays out
    print("Applying physical reflection of Ri on M1...")
    _, Ro = RTR_MT_M1_XY_input.calcRayIntersect(Ri, M, show_plot=False)
    # _, Ro = RTR_M1_XY_input.calcRayIntersect(Ri, M, show_plot=False)
    Ro = Ro.tolist()
    print("Dropping invalid rays of Ri...")
    Ri = physical_rays_drop(Ri, Ro)
    print("Done!")
    physical_data = generate_physical_data(Ri, Ro, M1_path)
    print("Summary table:")
    print(physical_data)

    # 4. save rays in, rays out and the mirror to PhysicalData directory
    M1_title = f"{M1_key}_{M1_name}" if M1_key is not None and len(M1_key) > 0 else f"{M1_name}"
    filename = f"{os.path.basename(rays_path).split('.')[0]}_{M1_title}"
    physical_data.to_csv(os.path.join(physical_data_dir, f"{filename}.csv"), index=False)

    # 5. add another mirror
    M2_path = os.path.join(base_dir, "Mirrors", "parabolic.pt")
    print(f"Loading Mirror M2 from: {M2_path}")
    M2 = torch.load(M2_path)
    print(f"Transforming Rays Ro to Ri#2 (Count={len(Ro)})")
    Ri2 = transform(Ro, kind="linear")
    print("Applying physical reflection of Ri#2 on M2...")
    _, Ro2 = RTR_MT_M2_YZ_input.calcRayIntersectM2(Ri2, M2, show_plot=False)
    # _, Ro2 = RTR_M2_YZ_input.calcRayIntersectM2(Ri2, M2, show_plot=False)
    Ro2 = Ro2.tolist()
    print("Dropping invalid rays of Ro#2...")
    Ri2 = physical_rays_drop(Ri2, Ro2)
    print("Done!")
    physical_data = generate_physical_data(Ri2, Ro2, M2_path)
    print("Summary table:")
    print(physical_data)

    # 6. save Data for two mirrors flow validation
    filename = f"{filename}_{os.path.basename(M2_path).split('.')[0]}"
    physical_data.to_csv(os.path.join(base_dir, "PhysicalData", f"{filename}.csv"), index=False)

    dt_object_end = current_time()
    print(f"Mirror Flow End Time = {dt_object_end}")
    print(f"Mirror Flow Duration (Sec) = {(dt_object_end - dt_object).total_seconds()}")


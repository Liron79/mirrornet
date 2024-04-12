import os
import torch
from utils import load_rays, physical_rays_drop, generate_physical_data, transform
from PhysicalScripts import RTR_M1_XY_input
from PhysicalScripts import RTR_M2_YZ_input


base_dir = os.path.dirname(os.path.abspath(__file__))
M1_name = "parabolic"
M1_dir = os.path.join(base_dir, "Mirrors")
rays_path = os.path.join(base_dir, "RaysIn", "pulse_1x2x1.csv")


if __name__ == "__main__":
    # 1. Load a mirror
    M1_path = os.path.join(M1_dir, f"{M1_name}.pt")
    M = torch.load(M1_path)
    print(f"Loading Mirror M1 from: {M1_path}")

    # 2. load rays in
    Ri = load_rays(path=rays_path)
    print(f"Loading Rays Ri (Count={len(Ri)}) from: {rays_path}")

    # 3. reflection - generate rays out
    print("Applying physical reflection of Ri on M1...")
    _, Ro = RTR_M1_XY_input.calcRayIntersect(Ri, M, show_plot=False)
    Ro = Ro.tolist()
    print("Dropping invalid rays of Ri...")
    Ri = physical_rays_drop(Ri, Ro)
    print("Done!")
    physical_data = generate_physical_data(Ri, Ro, M1_path)
    print("Summary table:")
    print(physical_data)

    # 4. save rays in, rays out and the mirror to PhysicalData directory
    filename = f"{os.path.basename(rays_path).split('.')[0]}_{os.path.basename(M1_path).split('.')[0]}"
    physical_data.to_csv(os.path.join(base_dir, "PhysicalData", f"{filename}.csv"), index=False)

    # 5. add another mirror
    M2_path = os.path.join(base_dir, "Mirrors", "parabolic.pt")
    print(f"Loading Mirror M2 from: {M2_path}")
    M2 = torch.load(M2_path)
    print(f"Transforming Rays Ro to Ri#2 (Count={len(Ro)})")
    Ri2 = Ro # transform(Ro, kind="linear") # TODO: Need to be validated mathematically!
    print("Applying physical reflection of Ri#2 on M2...")
    _, Ro2 = RTR_M2_YZ_input.calcRayIntersectM2(Ri2, M2, show_plot=False)
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
import os
import json
import pandas as pd
import torch
import numpy as np
from utils import load_rays, physical_rays_drop, generate_physical_data, gen_hash, transform
from PhysicalScripts import RTR_M1_XY_input
from PhysicalScripts import RTR_M2_YZ_input


base_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(base_dir, "PhysicalData")
data_list = [
    os.path.join(dir_path, "pulse_1x1x1_parabolic.csv")
]
mirror_dir = "5c3d14b758"
M1_dir = os.path.join(base_dir, "Mirrors", mirror_dir)
M1_name = "mirror"


if __name__ == "__main__":
    # 1. Loading Data from all test files
    Ri = list()
    Ro = list()
    for path in data_list:
        R = np.array(load_rays(path))
        Ri_ = R[:, :len(R[0]) // 2]
        Ro_ = R[:, len(R[0]) // 2:]
        Ri.extend(Ri_)
        Ro.extend(Ro_)

    # 2. Load the predicted mirror
    M1_path = os.path.join(M1_dir, f"{M1_name}.pt")
    M1 = torch.load(M1_path)
    print(f"Loading Predicted Mirror M1 from: {M1_path}")

    # 3. reflection - generate rays out using the predicted mirror
    print("Applying physical reflection of Ri on predicted M1...")
    _, Ro_pred = RTR_M1_XY_input.calcRayIntersect(Ri, M1, show_plot=False)
    Ro_pred = Ro_pred.tolist()
    print("Dropping invalid rays of Ri...")
    Ri_pred = physical_rays_drop(Ri, Ro_pred)
    print("Done!")
    physical_data_pred = generate_physical_data(Ri_pred, Ro_pred, M1_path)
    print("Summary table:")
    print(physical_data_pred)
    physical_data = pd.concat([pd.read_csv(data) for data in data_list], axis=0)

    physical_data["SE"] = pd.Series()
    physical_data_pred["SE"] = pd.Series()
    for index in np.unique(physical_data.Ro_ray_index.values):
        phy_df = physical_data[physical_data.Ro_ray_index == index][["Ro_x", "Ro_y", "Ro_z"]].reset_index(drop=True)
        phy_pred_idx = physical_data_pred.Ro_ray_index == index
        if phy_pred_idx.sum() == 0:
            continue
        phy_pred_df = physical_data_pred[physical_data_pred.Ro_ray_index == index][["Ro_x", "Ro_y", "Ro_z"]].reset_index(drop=True)
        physical_data_pred.loc[phy_pred_idx, ["SE"]] = (phy_df - phy_pred_df).pow(2).sum(axis=1).values.tolist()

    new_physical_data_pred = pd.concat((physical_data, physical_data_pred), axis=0)
    new_physical_data_pred = new_physical_data_pred.sort_values(by="Ro_ray_index")
    new_physical_data_pred.to_csv()

    # 4. save Data out + data analysis
    metadata = {
        "data_list": data_list,
        "MSE": physical_data_pred.SE.mean(),
        "num_physical_rays": len(physical_data),
        "num_physical_pred_rays": len(physical_data_pred),
        "num_rays_loss": len(physical_data) - len(physical_data_pred),
        "percentage_rays_loss": 1.0 - (len(physical_data_pred) / len(physical_data))
    }
    mirror_dir = os.path.join(dir_path, mirror_dir, gen_hash())
    os.makedirs(mirror_dir, exist_ok=True)

    physical_data_pred.to_csv(os.path.join(mirror_dir, "prediction.csv"), index=False)
    new_physical_data_pred.to_csv(os.path.join(mirror_dir, "full_data.csv"), index=False)
    with open(os.path.join(mirror_dir, "metadata.json"), "w+") as f:
        json.dump(metadata, f, indent=4)

    print(f"Prediction results directory: {mirror_dir}")
    print("Done!")
import os
import json
import pandas as pd
import torch
import numpy as np

from dataset import cell_sampling
from utils import load_rays, physical_rays_drop, generate_physical_data, gen_hash, current_time, load_json, transform
from PhysicalScripts import RTR_M1_XY_input, RTR_M2_YZ_input
from PhysicalScripts import RTR_MT_M1_XY_input, RTR_MT_M2_YZ_input


base_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(base_dir, "PhysicalData")
# data_path = os.path.join(dir_path, "pulse_1x2x2_parabolic.csv")
data_path = os.path.join(dir_path, "pulse_1x3x3_parabolic.csv")
# data_path = os.path.join(dir_path, "pulse_1x2x1_parabolic.csv")
# data_path = os.path.join(dir_path, "pulse_1x6x6_ebc16f8bbd_mirror.csv")

mirror_dir = "ebc16f8bbd"
M1_dir = os.path.join(base_dir, "Mirrors", mirror_dir)
# with open(os.path.join(M1_dir, "metadata.json")) as f:
#     cell_resolution = json.load(f)["cell_resolution"]
cell_resolution = 200
M1_name = "mirror"


if __name__ == "__main__":
    dt_object = current_time()
    print(f"Validation Flow Start Time = {dt_object}")

    # 1. Loading Data from all test files
    physical_data = list()
    Ri = list()
    Ro = list()
    R = pd.read_csv(data_path)
    R = cell_sampling(data=R, cell_resolution=cell_resolution)
    physical_data.append(R)
    R = R.loc[:, R.columns[1:]].values
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
    _, Ro_pred = RTR_MT_M1_XY_input.calcRayIntersect(Ri, M1, show_plot=False)
    Ro_pred = Ro_pred.tolist()
    print("Dropping invalid rays of Ri...")
    Ri_pred = physical_rays_drop(Ri, Ro_pred)
    print("Done!")
    physical_data_pred = generate_physical_data(Ri_pred, Ro_pred, M1_path)
    print("Summary table:")
    print(physical_data_pred)
    physical_data = pd.concat(physical_data, axis=0)

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

    # 4. save Data out + data analysis
    metadata = {
        "data_path": data_path,
        "cell_resolution": cell_resolution,
        "MSE": physical_data_pred.SE.mean(),
        "num_physical_rays": len(physical_data),
        "num_physical_pred_rays": len(physical_data_pred),
        "num_rays_loss": len(physical_data) - len(physical_data_pred),
        "percentage_rays_loss": 1.0 - (len(physical_data_pred) / len(physical_data))
    }
    mirror_dir = os.path.join(dir_path, mirror_dir, gen_hash())
    os.makedirs(mirror_dir, exist_ok=True)

    dt_object_end = current_time()
    print(f"Validation Flow End Time = {dt_object_end}")
    metadata["validation_time_sec"] = validation_time = (dt_object_end - dt_object).total_seconds()
    print(f"Validation Flow Duration(sec)= {validation_time}")

    physical_data_pred.to_csv(os.path.join(mirror_dir, "prediction.csv"), index=False)
    new_physical_data_pred.to_csv(os.path.join(mirror_dir, "full_data.csv"), index=False)
    with open(os.path.join(mirror_dir, "metadata.json"), "w+") as f:
        json.dump(metadata, f, indent=4)

    print(f"Prediction results directory: {mirror_dir}")
    print("Done!")
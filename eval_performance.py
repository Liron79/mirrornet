import os
import torch
import numpy as np
import pandas as pd
from utils import load_json

inputs = load_json("cfg/model_flow_exp_cfg.json")
M_ref = inputs["M_ref"]
M_pred = inputs["M_pred"]

ref_directory = inputs["ref_directory"]
pred_directory = inputs["pred_directory"]
data_ref = f"{inputs['data_basic_filename']}_{M_ref}"
data_pred = f"{inputs['data_basic_filename']}_{M_pred}"

base_dir = os.path.dirname(os.path.abspath(__file__))
mirror_dir_path = os.path.join(base_dir, "PhysicalMirrors")
phy_data_dir_path = os.path.join(base_dir, "Storage", "PhysicalData")

Z_r = torch.load(f"{os.path.join(mirror_dir_path, M_ref)}.pt")[-1]
Z_p = torch.load(f"{os.path.join(mirror_dir_path, M_pred)}.pt")[-1]

M_mse_loss = np.mean((Z_r - Z_p.numpy())**2)
print("MSE Loss @ Mirror[Optimum at 0]:", M_mse_loss)

D_r = pd.read_csv(os.path.join(phy_data_dir_path, ref_directory, data_ref, f"{data_ref}.csv"))[["Ro_ray_index", "Ro_z"]]
D_p = pd.read_csv(os.path.join(phy_data_dir_path, pred_directory, data_pred, f"{data_pred}.csv"))[["Ro_ray_index", "Ro_z"]]
D_r = D_r.set_index("Ro_ray_index")
D_p = D_p.set_index("Ro_ray_index")

Ro_mse_loss = ((D_r - D_p) / D_r).mean().Ro_z
print("MSE Loss @ Ro[Optimum at 0]:", Ro_mse_loss)

P = (D_r / D_r.sum()).values
Q = (D_p / D_p.sum()).values

epsilon = 1e-10
P = np.clip(P, epsilon, 1.0)
Q = np.clip(Q, epsilon, 1.0)

Ro_KL_div = np.sum(P * np.log(P / Q))
print("KL Div @ Ro[Optimum at 0]:", Ro_KL_div)
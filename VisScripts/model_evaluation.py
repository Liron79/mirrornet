import pandas as pd


Ro_parbolic_df = pd.read_csv(r"C:\Users\User\PycharmProjects\mirrornet\PhysicalData\customized_centered_rays_parabolic.csv")
Ro_model_df = pd.read_csv(r"C:\Users\User\PycharmProjects\mirrornet\PhysicalData\customized_centered_rays_d9f4223526_mirror.csv")

radius = 0.3
FP_y, FP_z = 0, 20

Ro_passed_rays_df = Ro_parbolic_df[(Ro_parbolic_df.Ro_y >= (FP_y - radius)) & (Ro_parbolic_df.Ro_y <= (FP_y + radius))]
Ro_passed_rays_df = Ro_passed_rays_df[(Ro_passed_rays_df.Ro_z >= (FP_z - radius)) & (Ro_passed_rays_df.Ro_z <= (FP_z + radius))]

Ro_model_passed_rays_df = Ro_model_df[Ro_model_df.index.isin(Ro_passed_rays_df.index)]
Ro_model_passed_rays_df = Ro_model_passed_rays_df[(Ro_model_passed_rays_df.Ro_y >= (FP_y - radius)) & (Ro_model_passed_rays_df.Ro_y <= (FP_y + radius))]
Ro_model_passed_rays_df = Ro_model_passed_rays_df[(Ro_model_passed_rays_df.Ro_z >= (FP_z - radius)) & (Ro_model_passed_rays_df.Ro_z <= (FP_z + radius))]

print("X Input Range:", Ro_parbolic_df.Ri_x.min(), Ro_parbolic_df.Ri_x.max())
print("Y Input Range:", Ro_parbolic_df.Ri_y.min(), Ro_parbolic_df.Ri_y.max())
print("Radius:", radius)
print("Parabolic:", len(Ro_passed_rays_df) / len(Ro_parbolic_df), f"{len(Ro_passed_rays_df)}/{len(Ro_parbolic_df)}")
print("Model:", len(Ro_model_passed_rays_df) / len(Ro_model_df), f"{len(Ro_model_passed_rays_df)}/{len(Ro_model_df)}")
print("Accuracy:", len(Ro_model_passed_rays_df) / len(Ro_passed_rays_df))
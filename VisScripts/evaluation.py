import pandas as pd


Ro_parbolic_df = pd.read_csv(r"C:\Users\User\PycharmProjects\mirrornet\PhysicalData\gaussian_parabolic.csv")
Ro_model_df = pd.read_csv(r"C:\Users\User\PycharmProjects\mirrornet\PhysicalData\gaussian_d9f4223526_mirror.csv")

Ro_parbolic_df["Ro_y_new"] = Ro_parbolic_df.Ro_y.round(0)
Ro_parbolic_df["Ro_z_new"] = Ro_parbolic_df.Ro_z.round(0)
Ro_parbolic_df = Ro_parbolic_df.loc[Ro_parbolic_df.Ro_y_new.isin([-1, 0, 1])&
                                    Ro_parbolic_df.Ro_z_new.isin([19, 20, 21])]

model_idx = list()
for i in Ro_parbolic_df.Ro_ray_index:
    model_idx.append(Ro_model_df.loc[Ro_model_df.Ro_ray_index == i].index[0])
Ro_model_df = Ro_model_df.loc[model_idx]

rel_error_y = abs(Ro_model_df.Ro_y.values - Ro_parbolic_df.Ro_y.values).mean()
rel_error_z = abs(Ro_model_df.Ro_z.values - Ro_parbolic_df.Ro_z.values).mean()

print("rel_error_y", rel_error_y)
print("rel_error_z", rel_error_z)
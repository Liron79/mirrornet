import os.path
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
physical_data_dir = os.path.join(base_dir, "Storage", "PhysicalData")

train_pred_path = os.path.join(physical_data_dir, "customized_rays_full_9fa846e6e6_mirror.csv")
val_pred_path = os.path.join(physical_data_dir, "customized_rays_full_275172cbff_mirror.csv")

train_path = os.path.join(physical_data_dir, "customized_rays_full_parabolicPFL0.5RFL50.csv")
val_path = os.path.join(physical_data_dir, "customized_rays_full_parabolicPFL0.5RFL48.csv")


def perf(gt_df: pd.DataFrame, pred_df: pd.DataFrame):
    gt_df_ = gt_df.loc[gt_df.Ro_ray_index.isin(pred_df.Ro_ray_index.unique().tolist())]
    gt_df_ = gt_df_.sort_values(by="Ro_ray_index")
    pred_df_ = pred_df.sort_values(by="Ro_ray_index")

    Lx = abs(pred_df_.Ro_x.values - gt_df_.Ro_x.values).mean()
    Ly = abs(pred_df_.Ro_y.values - gt_df_.Ro_y.values).mean()
    Lz = abs(pred_df_.Ro_z.values - gt_df_.Ro_z.values).mean()

    df = pd.DataFrame()
    df['gZ'] = gt_df_.Ro_z.round(3)
    df['pZ'] = pred_df_.Ro_z.round(3)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    sns.regplot(x='pZ', y='gZ', data=df)
    plt.show()

    return (Lx + Ly + Lz) / 3.0


if __name__ == "__main__":
    train_pred_df = pd.read_csv(train_pred_path)
    train_df = pd.read_csv(train_path)
    P_train = perf(gt_df=train_df, pred_df=train_pred_df)
    print("Training", f"{P_train=}")

    val_pred_df = pd.read_csv(val_pred_path)
    val_df = pd.read_csv(val_path)
    P_val = perf(gt_df=val_df, pred_df=val_pred_df)
    print("Validation", f"{P_val=}")
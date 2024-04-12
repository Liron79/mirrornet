import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(X: np.array, Y: np.array, Z: np.array, path: str, title: str = str(),
                 x_label: str = str(), y_label: str = str(), z_label: str = str(),
                 z_factor: float = 10**6, show: bool = False):
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z.T * z_factor, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel(x_label, fontsize=13, weight='semibold', rotation=-10)
    ax.set_ylabel(y_label, fontsize=13, weight='semibold', rotation=45)
    ax.set_zlabel(z_label, fontsize=13, weight='semibold', rotation=90)
    ax.yaxis._axinfo['label']['space_factor'] = 3.0
    if show:
        plt.show()
    plt.savefig(path)
import os
import numpy as np
import matplotlib.pyplot as plt


save = False
show = True
height = 128
width = 256
mean = (width / 4, height / 2)
sigma = 2


def generate_gaussian_beam(height, width, mean_x, mean_y, sigma):
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    amp = np.exp(-((x - mean_x)**2 + (y - mean_y)**2) / (2 * sigma**2))
    amp = amp[:, :, None]

    return x, y, amp


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir_path = os.path.join(base_dir, "Storage", "RaysOut")
    os.makedirs(dir_path, exist_ok=True)

    x, y, amp = generate_gaussian_beam(height, width, mean[0], mean[1], sigma)
    print(f"{amp.shape=}")

    # ax = plt.axes(projection='3d')
    # ax.plot_surface(x, y, amp[:, :, 0], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.set_title("Gaussian 3D Dist "f"{sigma=}")
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')

    plt.imshow(amp)
    plt.title("Gaussian 2D Dist "f"{sigma=}")
    plt.ylabel('Y')
    plt.xlabel('X')

    np.save(os.path.join(dir_path, f"gaussian_{int(mean[0])}_{int(mean[1])}_{int(sigma)}.npy"), amp)
    if show:
        plt.show()
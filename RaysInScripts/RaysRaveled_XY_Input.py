import numpy as np
import csv
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gaussian_dir_path = os.path.join(base_dir, "Storage", "RaysIn")
os.makedirs(gaussian_dir_path, exist_ok=True)


def gaussian_ft(x, k, sigma):
    """
    Calculates the Fourier transform of a Gaussian function.

    Args:
        x (float or numpy.ndarray): The value(s) of x at which to evaluate the Fourier transform.
        k (float or numpy.ndarray): The value(s) of k at which to evaluate the Fourier transform.
        sigma (float): The standard deviation of the Gaussian function.

    Returns:
        numpy.ndarray: The value(s) of the Fourier transform at the given x and k values.
    """
    norm = 1 / (sigma ** 2 * 2 * np.pi)
    exp_term = np.exp(-((k ** 2) * (sigma ** 4) + x ** 2) / (2 * (sigma ** 2)))
    return norm * exp_term


pulse_file_path = os.path.join(gaussian_dir_path, f"WigKx0Ky0_mode.csv")


def Raveled():
    with open(pulse_file_path, "w+", newline="") as csvfile:
        # with open('in_rays_M1.csv', 'w', newline='') as csvfile:
        columns = ["x", "y", "z", "kx", "ky", "kz", "ex", "ey", "ez", "distance", "amp", "status", "ray_index"]
        csv_writer = csv.writer(csvfile, delimiter=",")
        csv_writer.writerow(columns)
        ray_index = 0
        z = 60
        E = np.array([1, 0, 0])
        # kx = 1
        kz = -1
        with open('RaveledWigKx0Ky0pstackEx.csv', 'r') as file:
            reader = csv.reader(file)
            x = 40
            y = 2

            first_row = next(reader)
            second_row = next(reader)
            third_row = next(reader)

            j = np.size(first_row)

            for i in range(j):
                ray_index = ray_index + 1
                complex_number_x0 = first_row[i]  # Assuming the complex number is in the first column of each row
                complex_number_y0 = second_row[i]  # Assuming the complex number is in the first column of each row
                complex_number_amp = third_row[i]  # Assuming the complex number is in the first column of each row

                # Convert the string representation of the complex number to a complex number with float values
                complex_number_x0 = complex(complex_number_x0.replace('(', '').replace(')', ''))
                complex_number_y0 = complex(complex_number_y0.replace('(', '').replace(')', ''))
                complex_number_amp = complex(complex_number_amp.replace('(', '').replace(')', ''))

                real_part_x0 = complex_number_x0.real
                real_part_y0 = complex_number_y0.real
                real_part_amp = complex_number_amp.real

                kx = real_part_x0 / 10
                ky = real_part_y0 / 7

                amp = abs(real_part_amp)

                rayswriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                rayswriter.writerow([x, y, z, kx, ky, kz, E[0], E[1], E[2], 0, amp, 0, ray_index])
                # x0i, y0i, z0i, kxi, kyi, kzi, exi, eyi, ezi, distance, amplituda, status, ray_index


if __name__ == '__main__':
    Raveled()

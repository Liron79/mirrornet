import os
import numpy as np
import matplotlib.pyplot as plt
import csv


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pulse_dir_path = os.path.join(base_dir, "Storage", "RaysIn")
os.makedirs(pulse_dir_path, exist_ok=True)


n = 1
pulse_width = 7
pulse_length = 7
dim_x, dim_y, dim_z = 10, 10, 11


def plot_3d_to_2d(X, Y, Z, name='Plot'):
    plot_3d_to_2d.name += 1
    X, Y = np.meshgrid(X, Y, sparse=True)
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Kx", fontsize=13, weight='semibold', rotation=-10)
    ax.set_ylabel("Ky", fontsize=13, weight='semibold', rotation=45)
    ax.set_zlabel("W [mVs]", fontsize=13, weight='semibold', rotation=90)
    ax.yaxis._axinfo['label']['space_factor'] = 3.0
    plt.close()

plot_3d_to_2d.name = 0

def padding_function(temp_data, name="name"):
    origindata = np.copy(temp_data)
    zero_padding = np.zeros_like(origindata)
    padded_signal = np.concatenate((origindata, zero_padding))
    padded_signalroll = np.concatenate((zero_padding, origindata))
    padded_signal2 = np.concatenate((padded_signal.T, np.zeros_like(padded_signal.T))).T
    padded_signal2roll = np.concatenate((np.zeros_like(padded_signalroll.T), padded_signalroll.T)).T
    ########################################################## Padding CENTERED #######################################
    padded_signal_middle_roll = np.roll(padded_signal, len(padded_signal) // 2)
    padded_signal_middle_roll2 = np.concatenate((padded_signal_middle_roll.T, np.zeros_like(padded_signal_middle_roll.T))).T
    Pad_signal_mid_roll2 = np.roll(padded_signal_middle_roll2, len(origindata) // 2, axis=1)

    return padded_signal2, padded_signal2roll, Pad_signal_mid_roll2

########################################################################################################################
####################################### Main Program ###################################################################
########################################################################################################################

twodsquarewave123 = [[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]]

twodsquarewave1j = [[1, -1, 2, -1j, 1j],
                    [3, -3, 4, -3j, 3j],
                    [2, -2, 5, -2j, 1j],
                    [1,  2, 3,  4j, 5j],
                    [1,  2, 3, 4j, 5j]]

pulse_N = n
x, y = np.meshgrid(np.linspace(-n*dim_x, n*dim_y, n*dim_z), np.linspace(-n*dim_x, n*dim_y, n*dim_z))     # function space and parameters
twodsquarewave = np.where(abs(x) <= n*pulse_width, 1, 0) & np.where(abs(y) <= n*pulse_length, 1, 0)
origindata = np.copy(twodsquarewave)
E=origindata
print("E.shape :", E.shape)
Nx, Ny = E.shape
nx, ny= np.shape(E)
print("(Nx, Ny) :", Nx, Ny)
########################################################################################################################
a = 7.5*10**-3 # Length in m in x-direction
b = 5*10**-3   # Length in m in y-direction
dx = a/Nx      # step size in the x- and y- direction
dy = b/Ny
print("(dx, dy) \n", dx, dy)
x = np.linspace(-a*10**3 /2, a*10**3 /2, nx)
y = np.linspace(-b*10**3 /2, b*10**3 /2, ny)
X, Y = np.meshgrid(x, y)
print("(X, Y) \n", X.shape, Y.shape)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, abs(E*10**6).T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel("X [mm]", fontsize=13, weight='semibold')
ax.set_ylabel("Y [mm]", fontsize=13, weight='semibold')
ax.set_title(f"pulse_{pulse_N}x{pulse_width}x{pulse_length}")
area = "[mVs/m]"
decimals = 2
ax.set_zlabel("|Ex| {}\u00b2".format(area, decimals), fontsize=13, weight='semibold')
plt.show()

dx1, dy1 = 1, 1  # step size in the x- and y- direction
dkx = 2 * np.pi / (2 * Nx) / dx
dky = 2 * np.pi / (2 * Ny) / dy
print("(dkx, dky) \n", dkx, dky)
E_shift = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
W = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
Wig = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
A = np.zeros((2 * Nx, 2 * Ny), dtype=complex)
print("A \n", A.shape, E.shape)
B = np.zeros((2 * Nx, 2 * Ny), dtype=complex)
WigF = []
Etot = np.zeros((Nx, Ny, Nx, Ny), dtype=complex)
WigKx0Ky0 = np.zeros((Nx, Ny), dtype=complex)
WigKx0Ky0p = np.zeros((Nx, Ny), dtype=complex)
for i in range(Nx):
    for j in range(Ny):
        for m in range(-Nx, Nx, dx1):
            for n in range(-Ny, Ny, dy1):
                if (i - m) >= 0 and (j - n) >= 0 and (i + m) < Nx and (j + n) < Ny and (i - m) < Nx and (
                        j - n) < Ny and (i + m) >= 0 and (j + n) >= 0:
                    A[m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n])  # modified coorr matrix
                    E_shift[i, j, m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n])

        Exij = E_shift
        fft_Et = np.fft.fftshift(np.fft.fft2(Exij[i, j]))
        WigKx0Ky0[i, j] = fft_Et[Nx, Ny]
        Kx = np.linspace(-Nx, Nx, Exij[i, j].shape[0]) #* dkx
        Ky = np.linspace(-Ny, Ny, Exij[i, j].shape[1]) #* dky
        fx = np.fft.fftshift(np.fft.fftfreq(Kx.shape[0], Kx[1] - Kx[0]))
        fy = np.fft.fftshift(np.fft.fftfreq(Ky.shape[0], Ky[1] - Ky[0]))
        ##################################################################################################
        plot_3d_to_2d(fx, fy, np.abs(fft_Et), "Absolutute Amplitude Of the RaysIn")
        ##################################################################################################
        ########################### Padding Befor FFT ####################################################
        Exij_sl2, Exij_s2rl, Exij_m_rl2 = padding_function(E_shift[i, j], "Padded_Exij")  #
        Kxp = np.linspace(-Nx*8, Nx*8, Exij_m_rl2.shape[0]) * dkx
        Kyp = np.linspace(-Ny*8, Ny*8, Exij_m_rl2.shape[1]) * dky
        fxp = np.fft.fftshift(np.fft.fftfreq(Kxp.shape[0], Kx[1] - Kx[0]))
        fyp = np.fft.fftshift(np.fft.fftfreq(Kyp.shape[0], Ky[1] - Ky[0]))
        fft_Etp = np.fft.fftshift(np.fft.fft2(Exij_m_rl2))
        WigKx0Ky0p[i, j] = fft_Etp[fft_Etp.shape[0]//2, fft_Etp.shape[1]//2]
        #####################################################################################################
        # plot_3d_to_2d(fxp, fyp, np.abs(fft_Etp).T*10**6, "Absolutute Amplitude Of the RaysIn with padding")
        ######################################################################################################
        WigF.append(fft_Et)


x = np.linspace(-3.25, 3.25, WigKx0Ky0p.shape[0])
y = np.linspace(-2.5, 2.5, WigKx0Ky0p.shape[1])
X, Y = np.meshgrid(x, y)
Z = np.abs(WigKx0Ky0p*10**6).T
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', label='RaysIn Intensity')
ax.set_xlabel("X mm", fontsize=13, weight='semibold', rotation=-10)
ax.set_ylabel("Y mm", fontsize=13, weight='semibold', rotation=45)
ax.set_zlabel("W [mVs]", fontsize=13, weight='semibold', rotation=90)
ax.set_title(f"pulse_{pulse_N}x{pulse_width}x{pulse_length}_amp")
ax.yaxis._axinfo['label']['space_factor'] = 3.0
plt.show()

WigX0Y0p = WigF[len(WigF)//2]
print("WigX0Y0p.shape=\n", (WigX0Y0p.shape))
fig = plt.figure()
Kxp = np.linspace(-Nx*8, Nx*8, WigX0Y0p.shape[0]) * dkx
Kyp = np.linspace(-Ny*8, Ny*8, WigX0Y0p.shape[1]) * dky
print("Ex.shape[0] :", E.shape[0],"Exij.shape[0] :", Exij[i, j].shape[0], "WigX0Y0p.shape[0]:", WigX0Y0p.shape[0])
fxp = np.fft.fftshift(np.fft.fftfreq(Kxp.shape[0], Kx[1] - Kx[0]))
fyp = np.fft.fftshift(np.fft.fftfreq(Kyp.shape[0], Ky[1] - Ky[0]))
raveledfxp, raveledfyp, raveledWigX0Y0p = np.ravel(fxp), np.ravel(fyp), np.ravel(WigX0Y0p)
X, Y = np.meshgrid(fxp, fyp)

pulse_file_path = os.path.join(pulse_dir_path, f"pulse_{pulse_N}x{pulse_width}x{pulse_length}.csv")
with open(pulse_file_path, "w+", newline="") as f:
    columns = ["x", "y", "z", "kx", "ky", "kz", "ex", "ey", "ez", "distance", "amp", "status", "ray_index"]
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(columns)
    X_range = range(30, 52, 2)
    Y_range = range(-10, 12, 2)
    Z_const = 60
    TO_ENERGY = 10 ** 6
    idx = 1
    for ti, i in enumerate(X_range):
        for tj, j in enumerate(Y_range):
            t = Nx * ti + tj
            fft_Et_ij = np.abs(WigF[t]*TO_ENERGY).T
            for kx in range(Nx * 2):
                for ky in range(Ny * 2):
                    Ri = [i, j, Z_const, X[kx, ky], Y[kx, ky], -1, 1, 0, 0, 0, fft_Et_ij[kx, ky], 0, idx]
                    idx += 1
                    csv_writer.writerow(Ri)

#################################### Ravel #############################################################################
raveledX, raveledY, raveledWigX0Y0p = np.ravel(X), np.ravel(Y), np.ravel(WigX0Y0p)
print("raveledX, raveledY, raveledWigX0Y0p", raveledX.shape, raveledY.shape, raveledWigX0Y0p.shape)
WigX0Y0pstack = np.stack((raveledX, raveledY, raveledWigX0Y0p))
print("WigX0Y0pstack ", WigX0Y0pstack[0].shape, WigX0Y0p.shape)
reshapedWigX0Y0pstack = np.reshape(WigX0Y0pstack, (3, WigX0Y0p.shape[0], WigX0Y0p.shape[1]))
print("WigX0Y0pstack ", reshapedWigX0Y0pstack[0].shape, reshapedWigX0Y0pstack[1].shape,reshapedWigX0Y0pstack[2].shape)
print("TYPE WigX0Y0 ", type(X[0]), type(Y[2][3]), type(WigX0Y0p[1][3]))
print("TYPE WigX0Y0pstack ", type(reshapedWigX0Y0pstack[0]), type(reshapedWigX0Y0pstack[1][2][3]), type(reshapedWigX0Y0pstack[2][1][3]))
########################################################################################################################
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, np.abs(WigX0Y0p).T*10**6, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel("Kx", fontsize=13, weight='semibold', rotation=-10)
ax.set_ylabel("Ky", fontsize=13, weight='semibold', rotation=45)
ax.set_zlabel("W [mVs]", fontsize=13, weight='semibold', rotation=90)
ax.set_title(f"pulse_{pulse_N}x{pulse_width}x{pulse_length}_KxKy")
ax.yaxis._axinfo['label']['space_factor'] = 3.0
plt.show()
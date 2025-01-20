import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib


def plot_3d_to_2d(X, Y, Z, name='Plot'):
    Mxvalplt = np.abs(Z).max()
    # print("Maxvalue in plot function \n", Mxvalplt)
    Mxvalpltlist.append(Mxvalplt)
    # print("Number of Iteration \n", plot_3d_to_2d.name)
    plot_3d_to_2d.name += 1

    name_func = "figure" + str(plot_3d_to_2d.name)
    nf = "Ex.csv" + str(plot_3d_to_2d.name) + '.csv'
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.set_zlim(0, 0.0008)
    # ax.set_zlim(0, 0.5*10**-7)
    # ax.set_zlim(0, 1.0638514736070905e-07)
    # ax.set_zlim(0, 100)
    matplotlib.rc('xtick', labelsize=17)
    matplotlib.rc('ytick', labelsize=17)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # plt.title("Wigner X=0, Y=0", fontsize=15, weight='bold')
    # ax.set_xlabel("Kx", fontsize=25, weight='semibold', rotation=-10)
    # ax.set_ylabel("Ky", fontsize=25, weight='semibold', rotation=45)
    ax.set_xlabel("Kx", fontsize=15, weight='semibold')
    ax.set_ylabel("Ky", fontsize=15, weight='semibold')
    # ax.set_xlabel("Kx", fontsize=30, weight='bold', rotation=-10)
    # ax.set_ylabel("Ky", fontsize=30, weight='bold', rotation=45)
    # ax.set_zlabel(r'$\gamma$', fontsize=30, rotation=60)
    # ax.set_zlabel("W [mVs]", fontsize=13, weight='semibold', rotation=90)
    ax.set_zlabel("W [a.u.]", fontsize=17, weight='bold', rotation=90)
    ax.yaxis._axinfo['label']['space_factor'] = 3.0
    # xv, yv = np.meshgrid(X, Y)
    # plt.contour(xv, yv, Z, 100, cmap='viridis')
    # plt.savefig("C:/Users/michaelge/Desktop/WigPIc Ex and Ey/"+name_func+'.png')
    # plt.savefig("C:/Users/michaelge/Desktop/WigPIc Ex and Ey-Ey/"+name_func+'.png')
    # plt.savefig("C:/Users/michaelge/Desktop/ליהונתן/" + name_func + '.png')
    # plt.savefig("C:/Users/michaelge/Desktop/WIGPIC/WigPIc Ex and Ey Etp/" + name_func + '.png')
    # plt.savefig("C:/Users/michaelge/Desktop/WIGPIC/WIGPIC 6 10 21/" + name_func + '.png')
    # plt.savefig("C:/Users/michaelge/Desktop/WP140622/" + name_func + '.jpg')
    # np.savetxt("C:/Users/michaelge/Desktop/WP140622/" + nf, Z, delimiter=",")

    # plt.savefig("C:/Users/michaelge/Desktop/WDF2P/" + name_func + '.png')
    # plt.savefig("C:/Users/michaelge/Desktop/WofG/" + name_func + '.png')
    # plt.savefig("C:/Users/michaelge/Desktop/GDP31022/" + name_func + '.png')
    # plt.savefig("C:/Users/michaelge/Desktop/nowig/" + name_func + '.png')
    # plt.show()
    plt.close(fig)
    plt.close('all')
    # max(iterable, *iterables, key, default)
    # if Maxvalue >== max([Z])
    #     Maxvalue=max([Z])
    # print("Maxvalue \n", Maxvalue)
    return


plot_3d_to_2d.name = 0


def padding_function(temp_data, name="name"):
    origindata = np.copy(temp_data)
    # print("padded_signal origindata.shape \n", origindata.shape)
    ########################################################## Padding if necessery #######################################
    zero_padding = np.zeros_like(origindata)
    padded_signal = np.concatenate((origindata, zero_padding))
    padded_signalroll = np.concatenate((zero_padding, origindata))
    #
    # print("padded_signal \n", padded_signal)
    # print("padded_signalroll \n", padded_signalroll)
    padded_signal2 = np.concatenate((padded_signal.T, np.zeros_like(padded_signal.T))).T
    # print("padded_signal2 \n", padded_signal2)
    padded_signal2roll = np.concatenate((np.zeros_like(padded_signalroll.T), padded_signalroll.T)).T
    # print("padded_signal2roll \n", padded_signal2roll)
    ########################################################## Padding CENTERED #######################################
    padded_signal_middle_roll = np.roll(padded_signal, len(padded_signal) // 2)
    # print("padded_signal_middle_roll \n", padded_signal_middle_roll)
    padded_signal_middle_roll2 = np.concatenate(
        (padded_signal_middle_roll.T, np.zeros_like(padded_signal_middle_roll.T))).T
    # print("padded_signal_middle_roll2 \n", padded_signal_middle_roll2)
    Pad_signal_mid_roll2 = np.roll(padded_signal_middle_roll2, len(origindata) // 2, axis=1)
    # print("Pad_signal_mid_roll2 \n", Pad_signal_mid_roll2.shape)
    # print("Pad_signal_mid_roll2 \n", Pad_signal_mid_roll2)
    # raveledx, raveledy, raveledfunc = np.ravel(x), np.ravel(y), np.ravel(twodsquarewave)
    ################################################################
    return padded_signal2, padded_signal2roll, Pad_signal_mid_roll2


########################################################################################################################
####################################### Main Program ###################################################################
########################################################################################################################

######################################## Gaussian ######################################################################

dx = 0.1
fx = 10
x = np.arange(-0.5, 0.5, 1 / fx)
y = x
sigma = 0.1  # $\sigma$ of the gaussian
variance = sigma ** 2
X, Y = np.meshgrid(x, y)
# G=1/(np.sqrt(2*np.pi*variance))*(np.exp(-X**2/(2*variance)-Y**2/(2*variance))) # gaussian input signal
G2 = (np.exp(-X ** 2 / (2 * variance) - Y ** 2 / (2 * variance)))  # gaussian input signal
#
# ax = plt.axes(projection='3d')
# # ax.plot_surface(X, Y, abs(E*10**6).T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.plot_surface(X, Y, G2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# print("G2 \n", G2)
# plt.show()

#######################################################################################################################

# twodsquarewave123 = [[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 9]]
#
# twodsquarewave1j = [[1, -1, 2, -1j, 1j],
#                     [3, -3, 4, -3j, 3j],
#                     [2, -2, 5, -2j, 1j],
#                     [1,  2, 3,  4j, 5j],
#                     [1,  2, 3, 4j, 5j]]
n = 3
x, y = np.meshgrid(np.linspace(-n * 10, n * 10, n * 11),
                   np.linspace(-n * 10, n * 10, n * 11))  # function space and parameters
# print(abs(x) <= 5, 1, 0)
# print(y)
twodsquarewave = np.where(abs(x) <= n * 4, 1, 0) & np.where(abs(y) <= n * 3, 1, 0)

n = 3
a = 7.5 * 10 ** (-3)
b = 5 * 10 ** (-3)
# x = np.linspace(-a/2, a/2, WigKx0Ky0.shape[0])
# y = np.linspace(-b/2, b/2, WigKx0Ky0.shape[1])
# x, y = np.meshgrid(np.linspace(-n*a/2, n*a/2,  n*11), np.linspace(-n*b/2, n*b/2, n*11))     # function space and parameters


n = 3
x, y = np.meshgrid(np.linspace(-n * 10, n * 10, 3 * n * 11),
                   np.linspace(-n * 10, n * 10, 3 * n * 11))  # function space and parameters
x, y = np.meshgrid(np.linspace(-n * 10, n * 10, n * 11),
                   np.linspace(-n * 10, n * 10, n * 11))  # function space and parameters
# print(abs(x) <= 5, 1, 0)
print(max(x[0]) / 4)
# twodsquarewave =( ((np.where((x) >= -n*6, 1) & np.where((x) <= -n*3), 1) | ((np.where((x) >= n*3, 1) & np.where((x) <= n*6), 1)), 0 ) #& (np.where(abs(y) <= n*3, 1, 0)) #+ (np.where((x) >= -n*4, 1, 0) & np.where((x) <= -n*2, 1, 0)) & np.where(abs(y) <= n*3, 1, 0)
# twodsquarewave =( ((np.where( (x) >= -n*6, 1,0) & (x) <= -n*3), 1,0) | ((np.where((x) >= n*3, 1,0) & np.where((x) <= n*6), 1,0)) #& (np.where(abs(y) <= n*3, 1, 0))

rectwave2 = (np.where((x) < - 4, 1, 0) & np.where(abs(y) <= 3, 1, 0)) + (
            np.where((x) > 4, 1, 0) & np.where(abs(y) <= 1, 1, 0))

n = 3
a = 7.5
b = 5

x, y = np.meshgrid(np.linspace(-n * a / 2, n * a / 2, n * 11),
                   np.linspace(-n * b / 2, n * b / 2, n * 11))  # function space and parameters

n = 3
x, y = np.meshgrid(np.linspace(-n * 10, n * 10, n * 11),
                   np.linspace(-n * 10, n * 10, n * 11))  # function space and parameters

print(max(x[0]) / 4)

rectwave3 = (np.where((x) < - 4, 1, 0) & np.where(abs(y) <= 3, 1, 0)) + (
            np.where((x) > 4, 1, 0) & np.where(abs(y) <= 1, 1, 0))

# rectwave2 = (np.where((x) < - 4, 1, 0) & np.where((x) > - 10, 1, 0)) + np.where((x) > 4, 1, 0) & np.where((x) < 10, 1, 0) & (np.where(abs(y) <= 3, 1, 0))
# rectwave2 = (np.where((x) < - (max(x[0])-max(x[0])/2), 1, 0) & np.where((x) > - (max(x[0])-max(x[0])/2+2), 1, 0)) + np.where((x) > (max(x[0])-max(x[0])/2), 1, 0) & np.where((x) < (max(x[0])-max(x[0])/2+2), 1, 0) & (np.where(abs(y) <= 5, 1, 0))
# print("rectwave2 \n", rectwave2)
# rectwave2 = (np.where((x) < - (min(x[0])-min(x[0])/2), 1, 0) & np.where((x) > - (min(x[0])-min(x[0])/2+2), 1, 0)) + np.where((x) > (min(x[0])-min(x[0])/2), 1, 0) & np.where((x) < (min(x[0])-min(x[0])/2+2), 1, 0) & (np.where(abs(y) <= 3, 1, 0))
# print("rectwave \n", rectwave2)

rectwave4 = (np.where((x) < - (max(x[0]) / 16), 1, 0) & np.where((x) > - (max(x[0]) / 16 + 4), 1, 0)) + np.where(
    (x) > (max(x[0]) / 16), 1, 0) & np.where((x) < (max(x[0]) / 16 + 4), 1, 0) & (np.where(abs(y) <= 25, 1, 0))
# print("rectwave \n", rectwave2)

rectwave4complex = rectwave4 + 1j * rectwave4
# print("(Nx, Ny) \n", rectwave)
# print("rectwave \n", rectwave4)

# twodsquarewave = np.where(abs(x) <= n*3, 1, 0) & np.where(abs(y) <= n*3, 1, 0)
# twodsquarewave3 = np.where((x) > n*1 & (x) <= n*3), 1, 0) & np.where(abs(y) <= n*3, 1, 0)
# twodsquarewave3 = np.where((x) < -n*1 & (x) >= -n*3, 1) & np.where(abs(y) <= n*3, 1)

# twodsquarewave2 = np.where(((x)> 2) & (x <= n*4), 1, 0) & np.where((y) <= n*3, 1, 0) + np.where(((x)> -4) & (x <= n*(-2)), 1, 0) & np.where((y) <= n*3, 1, 0)
# print("(Nx, Ny) \n", twodsquarewave)
ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, abs(E*10**6).T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.plot_surface(x, y, rectwave4, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.plot_surface(x, y, twodsquarewave2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# plt.show()
######################################################################################################################
# data111 = loadmat('signals/Eqxtmat.mat')
data111 = loadmat('signals/Eqxt.mat')
data1111 = data111['Eqxt']
# dataEY = loadmat('Eqyt.mat')
# dataEy = dataEY['Eqyt']
origindata = np.copy(data1111)
# origindata = np.copy(dataEy)
# datasamemesh = loadmat('signals/Eqxtsamemesh.mat')
# datasmsh = datasamemesh['Eqxt']
# origindata = np.copy(datasmsh)
######################################################################################################################
# origindata = np.copy(twodsquarewave123)
# origindata = np.copy(twodsquarewave1j)
# origindata = np.copy(twodsquarewave)
# origindata = np.copy(twodsquarewave2)
# origindata = np.copy(rectwave2)
# origindata = np.copy(rectwave3)
# origindata = np.copy(rectwave4)
# origindata = np.copy(G2)
E = origindata
print("E.shape :", E.shape)
Nx, Ny = E.shape
nx, ny = np.shape(E)
print("(Nx, Ny) :", Nx, Ny)
########################################################################################################################
########################################################################################################################
a = 7.5 * 10 ** -3  # Length in m in x-direction
b = 5 * 10 ** -3  # Length in m in y-direction
dx = a / Nx  # step size in the x- and y- direction
dy = b / Ny
# print("(dx, dy) \n", dx, dy)
# xdimension, ydimension = Exfield.shape[0], Exfield.shape[1]
x = np.linspace(-a * 10 ** 3 / 2, a * 10 ** 3 / 2, nx)
y = np.linspace(-b * 10 ** 3 / 2, b * 10 ** 3 / 2, ny)
X, Y = np.meshgrid(x, y)
print("(X, Y) \n", X.shape, Y.shape)
matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)
ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, abs(E*10**6).T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.plot_surface(X, Y, abs(E*10**6), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, abs(E).T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel("X [mm]", fontsize=18, weight='semibold')
ax.set_ylabel("Y [mm]", fontsize=18, weight='semibold')
ax.set_title('twodsquarewave')
area = "[mVs/m]"
decimals = 2
# ax.set_zlabel("|Ex| {}\u00b2".format(area, decimals), fontsize=13, weight='semibold')
# ax.set_zlabel("|Ey| {}\u00b2".format(area, decimals), fontsize=13, weight='semibold')
ax.set_zlabel("2 Pulses [a.u.]".format(area, decimals), fontsize=18, weight='semibold')
# ax.set_zlabel("|Ex| [mVs/m]**2", fontsize=13, weight='semibold', rotation=90)
# ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.savefig("C:/Users/michaelge/Desktop/WigPIc Ex and Ey/"+name_func+'.png')
# plt.show()
np.savetxt("Ex.csv", E, delimiter=",")
######################################################################################################################
# matplotlib.rc('xtick', labelsize=20)
# matplotlib.rc('ytick', labelsize=20)

try:
    ax.plot_surface(X, Y, abs(E), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
except:
    ax.plot_surface(X, Y, abs(E).T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel("X [mm]", fontsize=13, weight='semibold')
ax.set_ylabel("Y [mm]", fontsize=13, weight='semibold')
ax.set_title('twodsquarewave')
area = "[mVs/m]"
decimals = 2
# ax.set_zlabel("|Ex| {}\u00b2".format(area, decimals), fontsize=13, weight='semibold')
ax.set_zlabel("G [a.u.]".format(area, decimals), fontsize=13, weight='semibold')
# ax.set_zlabel("|Ex| [mVs/m]**2", fontsize=13, weight='semibold', rotation=90)
# ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.savefig("C:/Users/michaelge/Desktop/WigPIc Ex and Ey/"+name_func+'.png')
# plt.show()
# np.savetxt("G.csv", E, delimiter=",")
# plot_3d_to_2d(nx,ny, np.abs(E), "Ex")
####################################################################################################################
# Exij_signal2, Exij_signal2roll, Exij_mid_roll2 = padding_function(origindata, "Padded_Exij") no necessery
#####################################################################################################################
itern = 0
# ExijL = []
ExijL = np.zeros((21 * Nx, 21 * Ny), dtype=complex)
L = 10
K = (2 * np.pi * 3 * 10 ** 12) / (3 * 10 ** 8)
print("(K) \n", K)
dx1, dy1 = 1, 1  # step size in the x- and y- direction
dkx = 2 * np.pi / (2 * Nx) / dx
dky = 2 * np.pi / (2 * Ny) / dy
# print("(dkx, dky) \n", dkx, dky)
E_shift = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
# W = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
W = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
Wig = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
# Wig2 = np.zeros((Nx, Ny, 4 * Nx, 4 * Ny), dtype=complex)
A = np.zeros((2 * Nx, 2 * Ny), dtype=complex)
print("A \n", A.shape, E.shape)
B = np.zeros((2 * Nx, 2 * Ny), dtype=complex)
WigF = []
Maxvalist = []
Mxvalpltlist = []
Etot = np.zeros((Nx, Ny, Nx, Ny), dtype=complex)
WigKx0Ky0 = np.zeros((Nx, Ny), dtype=complex)
WigKx0Ky0p = np.zeros((Nx, Ny), dtype=complex)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_path = os.path.join(base_dir, "Storage", "RaysIn")
os.makedirs(dir_path, exist_ok=True)
angle_thresh = 0.0001
output_file_path = os.path.join(dir_path, f"super_comp_3gauss_thresh{angle_thresh}.csv")
with open(output_file_path, "w+", newline="") as f:
    columns = ["x", "y", "z", "kx", "ky", "kz", "ex", "ey", "ez", "distance", "amp", "status", "ray_index"]
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(columns)
    Z = 60
    idx = 0
    for i in range(Nx):
        for j in range(Ny):
            for m in range(-Nx, Nx, dx1):
                for n in range(-Ny, Ny, dy1):
                    if (i - m) >= 0 and (j - n) >= 0 and (i + m) < Nx and (j + n) < Ny and (i - m) < Nx and (
                            j - n) < Ny and (i + m) >= 0 and (j + n) >= 0:
                        A[m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n])  # modified coorr matrix
                        E_shift[i, j, m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n])

            Exij = E_shift
            print(f"{Exij.shape=}")
            fft_Et = np.fft.fftshift(np.fft.fft2(Exij[i, j]))
            Maxvalue = fft_Et.max()
            Maxvalue = np.abs(fft_Et).max()
            Maxvalist.append(Maxvalue)
            WigKx0Ky0[i, j] = fft_Et[Nx, Ny]
            Kx = np.linspace(-Nx, Nx, Exij[i, j].shape[0])  # * dkx
            Ky = np.linspace(-Ny, Ny, Exij[i, j].shape[1])  # * dky
            Kx = np.linspace(-Nx * 10 ** (-3), Nx * 10 ** (-3), Exij[i, j].shape[0])  # * dkx
            Ky = np.linspace(-Ny * 10 ** (-3), Ny * 10 ** (-3), Exij[i, j].shape[1])  # * dky
            Kx = np.linspace(-Nx, Nx, Exij[i, j].shape[0])  # * dkx
            Ky = np.linspace(-Ny, Ny, Exij[i, j].shape[1])  # * dky
            Kx_normalized = Kx / K
            Ky_normalized = Ky / K
            fx = np.fft.fftshift(np.fft.fftfreq(Kx.shape[0], Kx[1] - Kx[0]))
            fy = np.fft.fftshift(np.fft.fftfreq(Ky.shape[0], Ky[1] - Ky[0]))
            print(f"{fx.shape=},{fy.shape=},{Kx.shape[0]=},{Kx[0]=},{Kx[1]=},{Kx[1] - Kx[0]=}")
            fx_normalized = np.fft.fftshift(np.fft.fftfreq(Kx_normalized.shape[0], Kx_normalized[1] - Kx_normalized[0]))
            fy_normalized = np.fft.fftshift(np.fft.fftfreq(Ky_normalized.shape[0], Ky_normalized[1] - Ky_normalized[0]))

            WigF.append(fft_Et)
            # angle_thresh = 0.02
            for k in range(len(fx)):
                for l in range(len(fy)):
                    xL = i + k * L
                    yL = j + l * L
                    # ExijL[xL, yL] += fft_Et[k, l]
                    kx = fx[k]
                    ky = fy[l]
                    amp = abs(fft_Et[k, l])
                    # print(k, l, kx, ky, amp)
                    # if kx == 0 and ky == 0:
                    #     Ri = [30 + i, -12 + j, Z, kx, ky, -1, 1, 0, 0, 0, amp, 0, idx]
                    #     idx += 1
                    #     csv_writer.writerow(Ri)
                    if abs(kx) <= angle_thresh >= abs(ky):
                        # Ri = [30 + i, -5 + j, Z, kx, ky, -1, 1, 0, 0, 0, amp, 0, idx]
                        Ri = [7 + i, -47 + j, Z, kx, ky, -1, 1, 0, 0, 0, amp, 0, idx]
                        idx += 1
                        csv_writer.writerow(Ri)

print("ExijL\n", abs(ExijL))
x = np.linspace(-100, 100, ExijL.shape[0])
y = np.linspace(-100, 100, ExijL.shape[1])

X, Y = np.meshgrid(x, y)
try:
    Z = np.abs(ExijL)
except:
    Z = np.abs(ExijL).T
fig = plt.figure()
ax = plt.axes(projection='3d')
try:
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', label='Rays Intensity')
except:
    ax.plot_surface(X, Y, Z.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none', label='Rays Intensity')
ax.set_xlabel("X mm", fontsize=22, weight='semibold', rotation=-10)
ax.set_ylabel("Y mm", fontsize=22, weight='semibold', rotation=45)
ax.set_zlabel("W [a.u.]", fontsize=23, weight='semibold', rotation=90)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.show()
plt.close('all')

indMaxList = np.argmax(Maxvalist)
Maxvalue1 = max(Maxvalist)

x = np.linspace(-3.25, 3.25, WigKx0Ky0p.shape[0])
y = np.linspace(-2.5, 2.5, WigKx0Ky0p.shape[1])

# X, Y = np.meshgrid(x, y)
# Z = np.abs(WigKx0Ky0p*10**6)#.T
#
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', label='Rays Intensity')
# ax.set_xlabel("X mm", fontsize=22, weight='semibold', rotation=-10)
# ax.set_ylabel("Y mm", fontsize=22, weight='semibold', rotation=45)
# ax.set_zlabel("W [a.u.]", fontsize=23, weight='semibold', rotation=90)
# ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.show()
# plt.close('all')


# #################################################### Ravel X Y  WigKx0Ky0p #############################################
# #
# raveledX, raveledY, raveledWigKx0Ky0p = np.ravel(X), np.ravel(Y), np.ravel(WigKx0Ky0p)
# print("raveledX, raveledY, raveledWigX0Y0p", raveledX.shape, raveledY.shape, raveledWigKx0Ky0p.shape)
# WigKx0Ky0pstack = np.stack((raveledX, raveledY, raveledWigKx0Ky0p))
# # # np.savetxt("RaveledWigKx0Ky0pstackPULS.csv", WigKx0Ky0pstack, delimiter=",")
# # # np.savetxt("RaveledWigKx0Ky0pstackEy.csv", WigKx0Ky0pstack, delimiter=",")
# np.savetxt("RaveledWigKx0Ky0pstackEx.csv", WigKx0Ky0pstack, delimiter=",")
# print("WigKx0Ky0pstack ", WigKx0Ky0pstack[0].shape, WigKx0Ky0p.shape)
# reshapedWigKx0Ky0pstack = np.reshape(WigKx0Ky0pstack, (3, WigKx0Ky0p.shape[0], WigKx0Ky0p.shape[1]))
# reshapedWigKx0Ky0pstack1 = np.reshape(WigKx0Ky0pstack, (3, int(np.sqrt(WigKx0Ky0pstack[0].shape[0])), int(np.sqrt(WigKx0Ky0pstack[0].shape[0]))))
# # print("WigKx0Ky0pstack ", reshapedWigKx0Ky0pstack[0].shape[0], reshapedWigKx0Ky0pstack[1].shape, reshapedWigKx0Ky0pstack[2].shape)
# # print("WigKx0Ky0pstack ", int(np.sqrt(WigKx0Ky0pstack[0].shape[0])), WigKx0Ky0pstack[1].shape, WigKx0Ky0pstack[1].shape)
# # print("reshapedWigKx0Ky0pstack1 ", reshapedWigKx0Ky0pstack1.shape)
# # print("reshapedWigKx0Ky0pstack111111111111111 ", int(np.sqrt(WigKx0Ky0pstack[0].shape[0])))
# #
# # print("reshapedWigKx0Ky0pstack1 ", type(reshapedWigKx0Ky0pstack1[0]), reshapedWigKx0Ky0pstack1[1], reshapedWigKx0Ky0pstack1[2])
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # ax = fig.gca(projection='3d')
# ax.plot_surface(np.real(reshapedWigKx0Ky0pstack1[0]), np.real(reshapedWigKx0Ky0pstack1[1]), np.abs(reshapedWigKx0Ky0pstack1[2]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.set_xlabel("X mm", fontsize=13, weight='semibold', rotation=-10)
# ax.set_ylabel("Y mm", fontsize=13, weight='semibold', rotation=45)
# ax.set_zlabel("WDF [mVs]", fontsize=13, weight='semibold', rotation=90)
# ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.show()
# plt.close('all')
# # np.savetxt("reshapedWigKx0Ky0pstack1Ey[2].csv", reshapedWigKx0Ky0pstack1[2], delimiter=",")
# ########################################## Max Value via /2 ############################################################
# # print("WigF=\n", (WigF[3].shape))
# # print("WigF=\n", (WigF[len(WigF)//2]).shape)
# # WigX0Y0p = WigF[len(WigF)//2]
# # # print("WigF=\n", WigX0Y0p)
# # # plot_3d_to_2d(np.linspace(-3.25, 3.25, WigX0Y0p.shape[0]), np.linspace(-2.5, 2.5, WigX0Y0p.shape[1]), np.abs(WigX0Y0p), "WigX0Y0p")
# # # print("WigX0Y0p.shape=\n", (WigX0Y0p.shape))
# # fig = plt.figure()
# # Kxp = np.linspace(-Nx*8, Nx*8, WigX0Y0p.shape[0]) * dkx
# # Kyp = np.linspace(-Ny*8, Ny*8, WigX0Y0p.shape[1]) * dky
# # print("Ex.shape[0] :", E.shape[0],"Exij.shape[0] :", Exij[i, j].shape[0], "WigX0Y0p.shape[0]:", WigX0Y0p.shape[0])
# # fxp = np.fft.fftshift(np.fft.fftfreq(Kxp.shape[0], Kx[1] - Kx[0]))
# # fyp = np.fft.fftshift(np.fft.fftfreq(Kyp.shape[0], Ky[1] - Ky[0]))
# # raveledfxp, raveledfyp, raveledWigX0Y0p = np.ravel(fxp), np.ravel(fyp), np.ravel(WigX0Y0p)
# # # print("raveledfxp, raveledfyp", raveledfxp.shape, raveledfyp.shape)
# # # X, Y = np.meshgrid(np.linspace(-3.75, 3.75, WigX0Y0p.shape[0]), np.linspace(-2.5, 2.5, WigX0Y0p.shape[1]))
# # X, Y = np.meshgrid(fxp, fyp)
# #
# # #################################### Ravel X Y  WigX0Y0p ###############################################################
# # raveledKx, raveledKy, raveledWigX0Y0p = np.ravel(X), np.ravel(Y), np.ravel(WigX0Y0p)
# # # print("raveledX, raveledY, raveledWigX0Y0p", raveledX.shape, raveledY.shape, raveledWigX0Y0p.shape)
# # WigX0Y0pstack = np.stack((raveledKx, raveledKy, raveledWigX0Y0p))
# # # np.savetxt("RaveledWigX0Y0pstackPULS.csv", WigX0Y0pstack, delimiter=",")
# # # np.savetxt("RaveledWigX0Y0pstackEx.csv", WigX0Y0pstack, delimiter=",")
# # # np.savetxt("RaveledWigX0Y0pstackEy.csv", WigX0Y0pstack, delimiter=",")
# # # print("WigX0Y0pstack ", WigX0Y0pstack[0].shape, WigX0Y0p.shape)
# # reshapedWigX0Y0pstack = np.reshape(WigX0Y0pstack, (3, WigX0Y0p.shape[0], WigX0Y0p.shape[1]))
# # reshapedWigX0Y0pstack1 = np.reshape(WigX0Y0pstack, (3, int(np.sqrt(WigX0Y0pstack[0].shape[0])), int(np.sqrt(WigX0Y0pstack[0].shape[0]))))
# # # print("WigX0Y0pstack ", reshapedWigX0Y0pstack[0].shape[0], reshapedWigX0Y0pstack[1].shape, reshapedWigX0Y0pstack[2].shape)
# # # print("WigX0Y0pstack ", int(np.sqrt(WigX0Y0pstack[0].shape[0])), WigX0Y0pstack[1].shape, WigX0Y0pstack[1].shape)
# # # print("reshapedWigX0Y0pstack1 ", reshapedWigX0Y0pstack1.shape)
# # #
# # # print("reshapedWigX0Y0pstack1 ", type(reshapedWigX0Y0pstack1[0]), reshapedWigX0Y0pstack1[1], reshapedWigX0Y0pstack1[2])
# #
# # # print("TYPE WigX0Y0 ", type(X[0]), type(Y[2][3]), type(WigX0Y0p[1][3]))
# # # print("TYPE WigX0Y0pstack ", type(reshapedWigX0Y0pstack[0]), type(reshapedWigX0Y0pstack[1][2][3]), type(reshapedWigX0Y0pstack[2][1][3]))
# # # ax.plot_surface(reshapedWigX0Y0pstack[0], reshapedWigX0Y0pstack[1], np.abs(reshapedWigX0Y0pstack[2]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# # ########################################################################################################################
# # ax = plt.axes(projection='3d')
# # # ax.plot_surface(X, Y, np.abs(WigX0Y0p).T*10**6, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# # # ax.plot_surface(X, Y, np.abs(WigX0Y0p)*10**6, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# # # ax.plot_surface(reshapedWigX0Y0pstack1[0][0], reshapedWigX0Y0pstack1[0][1], np.abs(reshapedWigX0Y0pstack1[2]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# # ax.plot_surface(np.real(reshapedWigX0Y0pstack1[0]), np.real(reshapedWigX0Y0pstack1[1]), np.abs(reshapedWigX0Y0pstack1[2]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# # print("TYPE WigX0Y0 ", type(reshapedWigX0Y0pstack1[0]), type(reshapedWigX0Y0pstack1[2][3]), type(reshapedWigX0Y0pstack1[1][3]))
# # # plt.title("Wigner X=0, Y=0", fontsize=15, weight='bold')
# # # ax.set_zlim(0, 0.08)
# # ax.set_xlabel("Kx", fontsize=13, weight='semibold', rotation=-10)
# # ax.set_ylabel("Ky", fontsize=13, weight='semibold', rotation=45)
# # # ax.set_zlabel(r'$\gamma$', fontsize=30, rotation=60)
# # ax.set_zlabel("W [mVs]", fontsize=13, weight='semibold', rotation=90)
# # ax.yaxis._axinfo['label']['space_factor'] = 3.0
# # # plt.ylim((10,30))
# # # plt.xlim((10,72))
# # # plt.zlim((10,30))
# # # plt.savefig("C:/Users/michaelge/Desktop/WIGPIC/"+name_func+'.png')
# #
# # # plt.show()
# #
# # # with open('WןignerEx.csv', 'wb') as csvfile:
# # #     filewriter = csv.writer(csvfile, delimiter=',',
# # #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
# # #     filewriter(fft_z)
# # #
# # # a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
# # # plot_3d_to_2d(fx, fy, np.abs(fft_Et), "Absolutute Amplitude Of the Rays")
# #
# # # np.savetxt("WigX0Y0p495Ex.csv", WigF[495], delimiter=",")
# # np.savetxt("WigKx0Ky0Ex.csv", WigKx0Ky0, delimiter=",")
# # # np.savetxt("WigKx0Ky0Puls.csv", WigKx0Ky0, delimiter=",")
# # # np.savetxt("WigX0Y0p527Ey.csv", WigF[527], delimiter=",")
# # # np.savetxt("WigKx0Ky0Ey.csv", WigKx0Ky0, delimiter=",")
# # # np.savetxt("WigX0Y0pPuls.csv", WigX0Y0p, delimiter=",")
# #

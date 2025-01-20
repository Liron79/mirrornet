import os

import numpy as np
from time import sleep, time
from scipy import signal
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import csv
from scipy.linalg import circulant
from scipy.fftpack import fft
import pandas as pd
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib

# matplotlib.use("Agg")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gaussian_dir_path = os.path.join(base_dir, "Storage", "RaysIn")
os.makedirs(gaussian_dir_path, exist_ok=True)


def plot_3d_to_2d(X, Y, Z, name='Plot'):
    Mxvalplt = np.abs(Z).max()
    # print("Maxvalue in plot function \n", Mxvalplt)
    Mxvalpltlist.append(Mxvalplt)
    print("Number of Iteration \n", plot_3d_to_2d.name)
    plot_3d_to_2d.name += 1

    name_func = "figure" + str(plot_3d_to_2d.name)
    nf = "Ex.csv" + str(plot_3d_to_2d.name) + '.csv'
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.set_zlim(0, 0.0008)
    # ax.set_zlim(0, 0.5*10**-7)
    # ax.set_zlim(0, 1.0638514736070905e-07)
    # ax.set_zlim(0, 1000)
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
    # plt.savefig(".../Desktop/WigPIc Ex and Ey/"+name_func+'.png')
    # plt.savefig(".../Desktop/WigPIc Ex and Ey-Ey/"+name_func+'.png')
    # plt.savefig(".../Desktop/ליהונתן/" + name_func + '.png')
    # plt.savefig(".../Desktop/WIGPIC/WigPIc Ex and Ey Etp/" + name_func + '.png')
    # plt.savefig(".../Desktop/WIGPIC/WIGPIC 6 10 21/" + name_func + '.png')
    # plt.savefig(".../Desktop/WP140622/" + name_func + '.jpg')
    # np.savetxt(".../Desktop/WP140622/" + nf, Z, delimiter=",")

    # plt.savefig(".../Desktop/WDPG/" + name_func + '.png')
    # plt.savefig(".../Desktop/WofG/" + name_func + '.png')
    # plt.savefig(".../Desktop/GDP31022/" + name_func + '.png')
    # plt.savefig(".../Desktop/nowig/" + name_func + '.png')
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
fx = 1
# x = np.arange(-0.5, 0.5, 1 / fx)
xdelta = 1
ydelta = 1
x = np.arange(-10 + xdelta, 12 + xdelta, 1 / fx)
# y = x
y = np.arange(-10 + ydelta, 12 + ydelta, 1 / fx)
sigma = 2  # $\sigma$ of the gaussian
variance = sigma ** 2
X, Y = np.meshgrid(x, y)
G = 1 / (np.sqrt(2 * np.pi * variance)) * (
    np.exp(-X ** 2 / (2 * variance) - Y ** 2 / (2 * variance)))  # gaussian input signal
G2 = (np.exp(-X ** 2 / (2 * variance) - Y ** 2 / (2 * variance)))  # gaussian input signal

# ax = plt.axes(projection='3d')
# # ax.plot_surface(X, Y, abs(E*10**6).T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.plot_surface(X, Y, G2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# # print("G2 \n", G2)
# # plt.show()


origindata = np.copy(G2)
E = origindata
# print("E :", E)
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

x = np.linspace(-a * 10 ** 3 / 2, a * 10 ** 3 / 2, nx)
y = np.linspace(-b * 10 ** 3 / 2, b * 10 ** 3 / 2, ny)
X, Y = np.meshgrid(x, y)
print("(X, Y) \n", X.shape, Y.shape)
matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, abs(E), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel("X [mm]", fontsize=15, weight='semibold')
ax.set_ylabel("Y [mm]", fontsize=15, weight='semibold')
area = "[mVs/m]"
decimals = 2
ax.set_zlabel("2 Pulses [a.u.]".format(area, decimals), fontsize=15, weight='semibold')
# ax.set_zlabel("|Ex| [mVs/m]**2", fontsize=13, weight='semibold', rotation=90)
# ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.savefig(".../Desktop/WigPIc Ex and Ey/"+name_func+'.png')
# plt.show()
# np.savetxt("Ex.csv", E, delimiter=",")
######################################################################################################################


ax.plot_surface(X, Y, abs(E), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel("X [mm]", fontsize=15, weight='semibold')
ax.set_ylabel("Y [mm]", fontsize=15, weight='semibold')
area = "[mVs/m]"
decimals = 2
# ax.set_zlabel("|Ex| {}\u00b2".format(area, decimals), fontsize=13, weight='semibold')
ax.set_zlabel("G [a.u.]".format(area, decimals), fontsize=15, weight='semibold')
# ax.set_zlabel("|Ex| [mVs/m]**2", fontsize=13, weight='semibold', rotation=90)
# ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.savefig(".../Desktop/WigPIc Ex and Ey/"+name_func+'.png')
# plt.show()
np.savetxt("G.csv", E, delimiter=",")
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
# Wig = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
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
for i in range(Nx):
    for j in range(Ny):
        for m in range(-Nx, Nx, dx1):
            for n in range(-Ny, Ny, dy1):
                if (i - m) >= 0 and (j - n) >= 0 and (i + m) < Nx and (j + n) < Ny and (i - m) < Nx and (
                        j - n) < Ny and (i + m) >= 0 and (j + n) >= 0:
                    A[m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n])  # modified coorr matrix
                    # print("A[m+Nx,n+Ny]=", A[m + Nx, n + Ny], "i=", i, "j=", j, "m=", m, "n=", n, m + Nx, n + Ny)
                    E_shift[i, j, m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n])
                    # print("E_shift[i,j,m+Nx,n+Ny]=", E_shift[i, j, m + Nx, n + Ny], "                  i=", i, "j=", j, "m=", m, "n=", n)
        # print("E_shift[i,j]=\n", E_shift)
        Exij = E_shift
        # ExijL = E_shift[i,j]
        # print("E_shift[i,j]=\n", Exij[i,j])
        # Exij[i,j,:,:] = E_shift
        # print("Exij[i, j] \n", Exij[i, j])
        fft_Et = np.fft.fftshift(np.fft.fft2(Exij[i, j]))
        # fft_Et = (Exij[i, j])
        # print("  i=", i, "j=", j, "fft_Et= \n", fft_Et)
        # print("fft_Et[i, j] =", fft_Et[i, j])
        # print("fft_Et[0, 0] =", fft_Et[Nx, Ny])
        Maxvalue = fft_Et.max()
        Maxvalue = np.abs(fft_Et).max()
        # print("Maxvalue \n", Maxvalue)
        Maxvalist.append(Maxvalue)
        WigKx0Ky0[i, j] = fft_Et[Nx, Ny]
        # print("WigKx0Ky0[i, j]  \n", WigKx0Ky0[i, j])
        # print("Exij[i, j] SHAPE \n", fft_Et[i, j].shape)
        # Efft.append(fft_Et)
        # Kx = np.linspace(-Nx // 2, Nx // 2, Exij.shape[0])  # * dkx
        # Ky = np.linspace(-Ny // 2, Ny // 2, Exij.shape[1])  # * dky
        Kx = np.linspace(-Nx, Nx, Exij[i, j].shape[0])  # * dkx
        Ky = np.linspace(-Ny, Ny, Exij[i, j].shape[1])  # * dky
        # Kx = np.linspace(-Nx * 10 ** (-3), Nx * 10 ** (-3), Exij[i, j].shape[0])  # * dkx - Liron
        # Ky = np.linspace(-Ny * 10 ** (-3), Ny * 10 ** (-3), Exij[i, j].shape[1])  # * dky - Liron
        Kx_normalized = Kx / K
        Ky_normalized = Ky / K
        # print("(kx, K) \n", Kx/K, Kx, K)
        # print("-Nx, Nx :", -Nx, Nx, "Kx,Ky \n", Kx, Ky)
        # print("Exij.shape[0] :", Exij[i, j].shape[0])
        # fft_z = np.fft.fftshift(np.fft.fft2(Exij))
        # Kx = np.linspace(-Nx, Nx, fft_z[i, j].shape[0]) * dkx
        # Ky = np.linspace(-Ny, Ny, fft_z[i, j].shape[1]) * dky
        # fft_zT = np.fft.fftshift(np.fft.fft2(Exij.T))
        fx = np.fft.fftshift(np.fft.fftfreq(Kx.shape[0], Kx[1] - Kx[0]))
        fy = np.fft.fftshift(np.fft.fftfreq(Ky.shape[0], Ky[1] - Ky[0]))
        fx_normalized = np.fft.fftshift(np.fft.fftfreq(Kx_normalized.shape[0], Kx_normalized[1] - Kx_normalized[0]))
        fy_normalized = np.fft.fftshift(np.fft.fftfreq(Ky_normalized.shape[0], Ky_normalized[1] - Ky_normalized[0]))
        fx_normalized = fx / K
        fy_normalized = fx / K
        # print("Kx[1] - Kx[0] \n", Kx[1], Kx[0], Kx[1] - Kx[0])
        # print("Kx,Ky \n", Kx, Ky)
        # print("fx,fy \n", fx, fy)
        # print("fx,fy \n", len(fx), fy)
        # ExijL[i, j] += Exij[math.floor(i+fx*L), math.floor(j+fy*L)]
        # print("fft_Et[i, j, :, :]shape \n", fft_Et[i,j].shape)
        # plot_3d_to_2d(fx, fy, np.abs(fft_z), "fft_Et")
        # plot_3d_to_2d(fx, fy, np.abs(fft_zT), "fft_Et Transposed")
        # print("fft_z[i, j, :, :] \n", fft_z)
        # print("Exij[i, j] \n", fft_Et.shape)
        ##################################################################################################
        # plot_3d_to_2d(fx, fy, np.abs(fft_Et), "Absolutute Amplitude Of the Rays")
        # plot_3d_to_2d(fx_normalized, fy_normalized, np.abs(fft_Et), "Absolutute Amplitude Of the Rays")

        # plot_3d_to_2d(fx, fy, np.real(fft_Et), "Absolutute Amplitude Of the Rays")
        ##################################################################################################
        ########################### Padding Befor FFT ####################################################
        Exij_sl2, Exij_s2rl, Exij_m_rl2 = padding_function(E_shift[i, j], "Padded_Exij")  #
        # Kxp = np.linspace(-Nx*8, Nx*8, Exij_m_rl2.shape[0]) * dkx
        # Kyp = np.linspace(-Ny*8, Ny*8, Exij_m_rl2.shape[1]) * dky
        # # print("Exij.shape[0] :", Exij[i, j].shape[0])
        # fxp = np.fft.fftshift(np.fft.fftfreq(Kxp.shape[0], Kx[1] - Kx[0]))
        # fyp = np.fft.fftshift(np.fft.fftfreq(Kyp.shape[0], Ky[1] - Ky[0]))
        fft_Etp = np.fft.fftshift(np.fft.fft2(Exij_m_rl2))
        WigKx0Ky0p[i, j] = fft_Etp[fft_Etp.shape[0]//2, fft_Etp.shape[1]//2]
        # print(fft_Etp[fft_Etp.shape[0]//2, fft_Etp.shape[1]//2], fft_Etp[fft_Etp.shape[0]//2+1, fft_Etp.shape[1]//2+1],fft_Etp[fft_Etp.shape[0]//2-1, fft_Etp.shape[1]//2-1])
        #####################################################################################################
        # print("fft_Etp[i, j].shape[0] :",i,j, fft_Etp.shape,Kxp.shape[0])
        # print("fft_Etp[i, j].shape[0] :", i, j, Exij_m_rl2.shape)
        ######################################################################################################
        # plot_3d_to_2d(fxp, fyp, np.abs(fft_Etp).T*10**6, "Absolutute Amplitude Of the Rays with padding")
        ######################################################################################################
        WigF.append(fft_Et)
        for k in range(len(fx)):
            for l in range(len(fy)):
                xL = i + k * L
                yL = j + l * L
                # print("xL, yL \n", k, l, xL, yL)
                ExijL[xL, yL] += fft_Et[k, l]
                # ExijL[math.floor(i + k * L)*100, math.floor(j + k * L)*100] += fft_Et[i,j]
                # ExijL[i, j] += Exij[math.floor(i + k * L), math.floor(j + k * L)]
                # print("ExijL[i, j] \n", ExijL[i, j], i,j)
                # print("[i, j] \n",  i, j,k,l, math.floor(i + k * L), math.floor(j + k * L), ExijL)
                # print("math.floor(i+fx*L) \n", math.floor(i + k*L)*100)
                # print(" Exij[math.floor(i + k * L), math.floor(j + k * L)] \n",  Exij[math.floor(i + k * L), math.floor(j + k * L)])
# Maxvalue1=max[WigF]
# Maxvalue=max([WigF])
# print("Maxvalue \n", Maxvalue)
# ind = np.argmax(WigF)
# inde = np.argmax[WigF]
# list(ind)
# print("ind \n", ind)
# print("(kx, K) \n", Kx/K, Kx, K)
# print("fx_normalized, fy_normalized, \n", fx_normalized, fy_normalized)
# print("ExijL\n", abs(ExijL))
x = np.linspace(-100, 100, ExijL.shape[0])
y = np.linspace(-100, 100, ExijL.shape[1])
# x = np.linspace(-a/2, a/2, WigKx0Ky0.shape[0])
# y = np.linspace(-b/2, b/2, WigKx0Ky0.shape[1])
# print(Nx,Ny, x,y)
X, Y = np.meshgrid(x, y)
Z = np.abs(ExijL)  # .T

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z / np.abs(Z).max(), rstride=1, cstride=1, cmap='viridis', edgecolor='none',
                label='Rays Intensity')
# ax.set_xlabel("X mm", fontsize=15, weight='semibold', rotation=-10)
# ax.set_ylabel("Y mm", fontsize=15, weight='semibold', rotation=45)
ax.set_xlabel("X [mm]", fontsize=15, weight='semibold')
ax.set_ylabel("Y [mm]", fontsize=15, weight='semibold')
# ax.set_zlabel("W [mVs]", fontsize=13, weight='semibold', rotation=90)
ax.set_zlabel("W [a.u.]", fontsize=15, weight='semibold', rotation=90)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.show()
plt.close('all')

# print("MaxList495 \n", Maxvalist[495])
#
# print("MaxList497 \n", Maxvalist[497])
# print("MaxList529 \n", Maxvalist[529])
indMaxList = np.argmax(Maxvalist)
# print("indMaxList \n", indMaxList)
Maxvalue1 = max(Maxvalist)
# print("Maxvalue1 \n", Maxvalue1)
# print("MaxvalueMatrix \n", WigF[527])

# indMxvalpltlist = np.argmax(Mxvalpltlist)
# print("Mxvalpltlist \n", indMxvalpltlist)
# Mxvalpltlist1 = max(Mxvalpltlist)
# print("Mxvalpltlist1 \n", Mxvalpltlist1)
# # print("Mxvalpltlist495 \n", Mxvalpltlist[495])

# print("WigKx0Ky0=\n", WigKx0Ky0)
x = np.linspace(-3.25, 3.25, WigKx0Ky0p.shape[0])
y = np.linspace(-2.5, 2.5, WigKx0Ky0p.shape[1])
# x = np.linspace(-a/2, a/2, WigKx0Ky0.shape[0])
# y = np.linspace(-b/2, b/2, WigKx0Ky0.shape[1])
# print(Nx,Ny, x,y)
X, Y = np.meshgrid(x, y)
# Z = np.abs(WigKx0Ky0p*10**6)#.T
Z = np.abs(WigKx0Ky0)  # .T
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z / np.abs(Z).max(), rstride=1, cstride=1, cmap='viridis', edgecolor='none',
                label='Rays Intensity')
ax.set_xlabel("X mm", fontsize=22, weight='semibold', rotation=-10)
ax.set_ylabel("Y mm", fontsize=22, weight='semibold', rotation=45)
# ax.set_zlabel("W [mVs]", fontsize=13, weight='semibold', rotation=90)
ax.set_zlabel("W [a.u.]", fontsize=23, weight='semibold', rotation=90)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.savefig(".../Desktop/WDPG/" + 'GAUSS1000mm' + '.png')
# plt.show()
plt.close('all')

# plot_3d_to_2d(x, y, np.abs(WigKx0Ky0), "WigKx0Ky0")

# #################################################### Ravel X Y  WigKx0Ky0p #############################################
#
# raveledX, raveledY, raveledWigKx0Ky0p = np.ravel(X), np.ravel(Y), np.ravel(WigKx0Ky0p)
# print("raveledX, raveledY, raveledWigX0Y0p", raveledX.shape, raveledY.shape, raveledWigKx0Ky0p.shape)
# WigKx0Ky0pstack = np.stack((raveledX, raveledY, raveledWigKx0Ky0p))
# # np.savetxt("RaveledWigKx0Ky0pstackPULS.csv", WigKx0Ky0pstack, delimiter=",")
# # np.savetxt("RaveledWigKx0Ky0pstackEy.csv", WigKx0Ky0pstack, delimiter=",")
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
# # plt.show()
# plt.close('all')
# # np.savetxt("reshapedWigKx0Ky0pstack1Ey[2].csv", reshapedWigKx0Ky0pstack1[2], delimiter=",")

WigX0Y0p = WigF[len(WigF)//2]
print("WigX0Y0p.shape=\n", WigX0Y0p.shape)
fig = plt.figure()
Kxp = np.linspace(-Nx*8, Nx*8, WigX0Y0p.shape[0]) * dkx
Kyp = np.linspace(-Ny*8, Ny*8, WigX0Y0p.shape[1]) * dky
print("Ex.shape[0] :", E.shape[0],"Exij.shape[0] :", Exij[i, j].shape[0], "WigX0Y0p.shape[0]:", WigX0Y0p.shape[0])
fxp = np.fft.fftshift(np.fft.fftfreq(Kxp.shape[0], Kx[1] - Kx[0]))
fyp = np.fft.fftshift(np.fft.fftfreq(Kyp.shape[0], Ky[1] - Ky[0]))
raveledfxp, raveledfyp, raveledWigX0Y0p = np.ravel(fxp), np.ravel(fyp), np.ravel(WigX0Y0p)
X, Y = np.meshgrid(fxp, fyp)

pulse_file_path = os.path.join(gaussian_dir_path, f"mode1.csv")
with open(pulse_file_path, "w+", newline="") as f:
    columns = ["x", "y", "z", "kx", "ky", "kz", "ex", "ey", "ez", "distance", "amp", "status", "ray_index"]
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(columns)
    Z_const = 60
    TO_ENERGY = 10 ** 6
    idx = 1
    print(f"{X.shape=}")
    for ti, i in enumerate(np.arange(30, 52, 1)):
        for tj, j in enumerate(np.arange(-10, 12, 1)):
            t = Nx * ti + tj
            fft_Et_ij = np.abs(WigF[t])
            for kx in range(Nx * 2):
                for ky in range(Ny * 2):
                    # if X[kx, ky] == 0 and Y[kx, ky] == 0: # par
                    # if X[kx, ky] != 0 and Y[kx, ky] != 0: # non-par
                    Ri = [i, j, Z_const, X[kx, ky], Y[kx, ky], -1, 1, 0, 0, 0, fft_Et_ij[ti, tj], 0, idx]
                    idx += 1
                    csv_writer.writerow(Ri)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift

dx = 0.1
# fx = 80
fx = 8
x = np.arange(-5, 5, 1 / fx)
y = x
sigma = 0.1  # $\sigma$ of the gaussian
variance = sigma ** 2
X, Y = np.meshgrid(x, y)
G = 1 / (np.sqrt(2 * np.pi * variance)) * (np.exp(-X ** 2 / (2 * variance) - Y ** 2 / (2 * variance)))  # gaussian input signal
G2 = (np.exp(-X ** 2 / (2 * variance) - Y ** 2 / (2 * variance)))  # gaussian input signal

ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, abs(E*10**6).T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, G2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title("Gaussian Input")
plt.show()

ax = plt.axes(projection='3d')
Kx, Ky = X, Y
fft_G = np.fft.fftshift(np.fft.fft2(G2))
fx = np.fft.fftshift(np.fft.fftfreq(Kx.shape[0], Kx[1] - Kx[0]))
fy = np.fft.fftshift(np.fft.fftfreq(Ky.shape[0], Ky[1] - Ky[0]))
ax.plot_surface(Kx, Ky, abs(fft_G), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.plot_surface(Kx, Ky, abs(np.fft.fft2(G2)), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
plt.show()
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize = (5,8))
# # subplots_adjust(hspace=0.75)
#
# ax1.plot(X, Y, G2, color = "black", linewidth = 1, linestyle = "--");
# ax1.set_title('Gaussian Pulse $\sigma$={}'.format(sigma))
# ax1.set_xlabel('Time(s)')
# ax1.set_ylabel('Amplitude')
#
# ax2.plot(X, Y, G2, color = "black", linewidth = 1, linestyle = "--", label =             "Scipy")
# # ax2.plot(f, Xtheory, color = "blue",linewidth = 5, alpha = 0.25, label = "Theory")
# ax2.set_title('Magnitude of FFT');
# ax2.set_xlabel('Frequency (Hz)')
# ax2.set_ylabel('|X(f)|');
# ax2.set_xlim(-10,10)
# ax2.legend()
#
# plt.show()

fs = 80  # sampling frequency
fs = 8
t = np.arange(-0.5, 0.5, 1 / fs)  # time domain
sigma = 0.1  # $\sigma$ of the gaussian
variance = sigma**2

x = 1 / (np.sqrt(2 * np.pi * variance)) * (np.exp(-t ** 2 / (2 * variance)))  # gaussian input signal

L = len(x)
NFFT = 1024  # length of FFT
f = (fs / NFFT) * np.arange(-NFFT / 2, NFFT / 2)  # frequency domain
X = fftshift(fft(x, NFFT))  # FFT of the gaussian signal
# X = (fft(x,NFFT))
Xtheory = np.exp(-0.5 * (2 * np.pi * sigma * f) ** 2)  # theoretical FFT of the gaussian signal

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 8))
# subplots_adjust(hspace=0.75)

ax1.plot(t, x, color="black", linewidth=1, linestyle="--")
ax1.set_title('Gaussian Pulse $\sigma$={}'.format(sigma))
ax1.set_xlabel('Time(s)')
ax1.set_ylabel('Amplitude')

ax2.plot(f, abs(X) / L, color="black", linewidth=1, linestyle="--", label="Scipy")
ax2.plot(f, Xtheory, color="blue", linewidth=5, alpha=0.25, label="Theory")
ax2.set_title('Magnitude of FFT')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('|X(f)|')
ax2.set_xlim(-10, 10)
ax2.legend()

plt.show()

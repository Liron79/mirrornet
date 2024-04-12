from PhysicalScripts.spline import *
import matplotlib.pyplot as plt
import math


def coordinates(x0, y0, z0, kxi,  kyi, kzi, t):
    x = x0 + t * kxi
    y = y0 + t * kyi
    z = z0 + t * kzi

    return x, y, z


def find_start(x, y, m, n, x0i, y0i, z0i, kxi,  kyi, kzi, t_ref, MLm, status):
    spl = ourspline(x0i, y0i, x, y, m, n, MLm, 1, status)
    if z0i < spl[0]: #LM
        t_ref = (spl - z0i + 1) / kzi
        x0i, y0i, z0i = coordinates(x0i, y0i, z0i, kxi,  kyi, kzi, t_ref)

    return x0i, y0i, z0i

def spline_mirror(x, y, z):

    m = np.size(x)
    n = np.size(y)
    x, y = np.meshgrid(x, y)
    z = np.reshape(z, (m, n), order='F')

    return x, y, z


# distance represents the magnitude of the vector from the ray's origin to the impact point
def raydistance(x0i, y0i, z0i, xx0i, yy0i, zz0i):

    L = math.sqrt((x0i - xx0i)**2 + (y0i- yy0i)**2 + (z0i-zz0i)**2)

    return L


def plot_3d_to_2d(X, Y, Z, name='Plot'):
    X, Y = np.meshgrid(X, Y)
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_title(name)

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    # xv, yv = np.meshgrid(X, Y)
    # plt.contour(xv, yv, Z, 100, cmap='viridis')
    plt.show()
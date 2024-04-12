import numpy as np


def get_mesh1(M1x, M1y, M1z):
    # type of boundary condition
    ibx = 2
    iby = 2

    m = np.size(M1x)
    x = np.array(M1x)

    n = np.size(M1y)
    y = np.array(M1y)

    z = np.array(M1z)

    z = z.reshape(n*m)

    zl = np.zeros(n)

    zr = np.zeros(n)
    zu = np.zeros(m)
    zd = np.zeros(m)
    MLm = np.zeros(m * n * 16)

    return m, x, n, y, z, zl, zr, zu, zd, ibx, iby, MLm


def get_mesh2(M2x, M2y, M2z):
    # type of boundary condition
    ibx = 2
    iby = 2

    m = np.size(M2x)
    x = np.array(M2x)

    n = np.size(M2y)
    y = np.array(M2y)

    z = np.array(M2z)

    z = z.reshape(n*m)

    zl = np.zeros(n)

    zr = np.zeros(n)
    zu = np.zeros(m)
    zd = np.zeros(m)
    MLm = np.zeros(m * n * 16)

    return m, x, n, y, z, zl, zr, zu, zd, ibx, iby, MLm
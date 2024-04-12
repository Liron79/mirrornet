import math
from PhysicalScripts.spline import *
from PhysicalScripts.get_mesh import *
from datetime import datetime
from PhysicalScripts.helper import coordinates, find_start, spline_mirror, raydistance
import matplotlib.pyplot as plt
from PhysicalScripts import spline_calculation


def ray_tracing(x0i, y0i, z0i, kxi,  kyi, kzi, exi, eyi, ezi, status, x, y, m, n, MLm):
    K = np.array([[kxi, kyi, kzi]])
    E = np.array([[exi, eyi, ezi]])

    K = K.transpose()
    E = E.transpose()

    K = K / math.sqrt(K[0] **2 + K[1]**2 + K[2]**2)  # normalized

    k0 = math.sqrt(kxi**2 + kyi**2 + kzi**2)  # 7

    tolerance = 0.000005  # tolerance in units of wavelength

    # tmin = x[0] / K[0]
    # tmax = x[m - 1] / K[0]
    # tmid = (tmax - tmin) / 2 + tmin

    tmin = 0
    tmax = -z0i / K[2]  # check intersection
    tmid = (tmax - tmin) / 2

    xx, yy, zz = coordinates(x0i, y0i, z0i, kxi,  kyi, kzi, tmid)
    xx = float(xx)
    yy = float(yy)
    zz = float(zz)

    zspl, Nx, Ny, Nz, status = ourspline(xx, yy, x, y, m, n, MLm, 1, status)
    count = 1000
    if status != 0:
        X = np.zeros(3)
        K = np.zeros(3)
        E = np.zeros(3)
        return X, K, E, 1

    while (np.abs(zz-zspl) >= 2 * np.pi / k0 * tolerance) and count > 0:
        if zz > zspl:
            tmin = tmid
            tmid = (tmax - tmid) / 2 + tmin
        else:
            tmax = tmid
            tmid = (tmid - tmin) / 2 + tmin

        xx, yy, zz = coordinates(x0i, y0i, z0i, kxi, kyi, kzi, tmid)

        zspl, Nx, Ny, Nz, status = ourspline(xx, yy, x, y, m, n, MLm, 1,status)
        if status != 0:
            X = np.zeros(3)
            K = np.zeros(3)
            E = np.zeros(3)
            return X, K, E, 1
        count = count - 1

    if count == 0:
        X = np.zeros(3)
        K = np.zeros(3)
        E = np.zeros(3)
        return X, K, E, 1

    spl, Nx, Ny, Nz ,status = ourspline(xx, yy, x, y, m, n, MLm, 0,status)
    if status != 0:
        X = np.zeros(3)
        K = np.zeros(3)
        E = np.zeros(3)
        return X, K, E, 1

    N = [Nx[0], Ny[0], Nz[0]]
    N = np.asarray(N)
    NN = math.sqrt(Nx**2 + Ny**2 + Nz**2)
    N =N / NN
    X =[xx, yy, zz]

    TR = sum(N * K)

    K = K - 2 * TR * N
    # E = 2 * (N.dot(E) ) * N - E
    TR = sum(N * E)
    E = 2 * TR * N - E
    return X, K, E, status


def RT(x0i, y0i, z0i, kxi,  kyi, kzi, exi, eyi, ezi, length, amp, status, ray_index, SplineParam):

    x = SplineParam[0]
    y = SplineParam[1]
    MLm = SplineParam[2]

    m = np.size(x)
    n = np.size(y)


    x0i, y0i, z0i = find_start(x, y, m, n, x0i, y0i, z0i, kxi, kyi, kzi, 0, MLm, status)
    if status == 0:
        [X, K, E, status] = ray_tracing(x0i, y0i, z0i, kxi,  kyi, kzi, exi, eyi, ezi, status, x, y, m, n, MLm)
        M = [float(X[0]), float(X[1]), float(X[2]), float(K[0]), float(K[1]), float(K[2]), float(E[0]), float(E[1]),
             float(E[2]), float(length), float(amp), int(status), int(ray_index)]
    else:
        M =[x0i, y0i, z0i, kxi,  kyi, kzi, exi, eyi, ezi, length, amp, status, ray_index]

    return M


def calcRayIntersect(Ri, M1, show_plot:bool = True):
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = datetime.fromtimestamp(timestamp)
    if show_plot:
        print("Start =", dt_object)
    plt.figure(1)
    plt.clf()

    M1x = M1[0]
    M1y = M1[1]
    M1z = M1[2]
    SplineParam = spline_calculation.SplineCalculation_M1(M1x, M1y, M1z)

    Rint = []
    Ro = []
    for i in Ri:
        x0i = i[0]
        y0i = i[1]
        z0i = i[2]
        kxi = i[3]
        kyi = i[4]
        kzi = i[5]
        exi = i[6]
        eyi = i[7]
        ezi = i[8]
        distance = i[9]
        amp = i[10]
        status = int(i[11])
        ray_index = int(i[12])

        M = RT(x0i, y0i, z0i, kxi, kyi, kzi, exi, eyi, ezi, distance, amp, status, ray_index, SplineParam)

        L1 = raydistance(x0i, y0i, z0i, M[0], M[1], M[2])  # calculate raylength start-intersection
        M[9]= distance + L1

        Rint.append(M)
        if M[3] != 0:
                T = -M[0] / M[3]  #  t=-x0/kx (x=0)
                xx0i, yy0i, zz0i = coordinates(M[0], M[1], M[2], M[3], M[4], M[5], T)
                L2 = raydistance(M[0], M[1], M[2], xx0i, yy0i, zz0i)
                Mx = xx0i, yy0i, zz0i, M[3], M[4], M[5], M[6], M[7], M[8], M[9]+L2, amp, status, ray_index
                Ro.append(Mx) # ray at focal point

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object_end = datetime.fromtimestamp(timestamp)
    print("Finish:", dt_object_end)
    print("Total-Time:", dt_object_end - dt_object)

    if show_plot:
        plt.show()

    return np.array(Rint), np.array(Ro)
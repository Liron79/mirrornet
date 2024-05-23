import math
import multiprocessing as mp
import os
from PhysicalScripts.spline import *
from PhysicalScripts.get_mesh import *
from datetime import datetime
from PhysicalScripts.helper import coordinates,find_start,spline_mirror,raydistance
import matplotlib.pyplot as plt
from PhysicalScripts import spline_calculation
from utils import current_time


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
    #
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
    #E = 2 * (N.dot(E) ) * N - E
    TR = sum(N * E)
    E = 2 * TR * N - E
    return X, K, E, status


def RT(myTuple, SplineParam, result_list, lock):
    process_id = os.getpid()
    print("Process", process_id, "is running.")
    M_final = []
    Ro = []
    x = SplineParam[0]
    y = SplineParam[1]
    MLm = SplineParam[2]

    m = np.size(x)
    n = np.size(y)
    for i in myTuple:
        x0i, y0i, z0i, kxi, kyi, kzi, exi, eyi, ezi, length, amp, status, ray_index = i

        if status == 0:
            [X, K, E, status] = ray_tracing(x0i, y0i, z0i, kxi,  kyi, kzi, exi, eyi, ezi, status, x, y, m, n, MLm)
            M = [float(X[0]), float(X[1]), float(X[2]), float(K[0]), float(K[1]), float(K[2]), float(E[0]),
                     float(E[1]),float(E[2]), float(length), float(amp), int(status), int(ray_index)]
            L1 = raydistance(x0i, y0i, z0i, M[0], M[1], M[2])
            M[9] = length + L1
            M_final.append(M)
        else:
            M =[x0i, y0i, z0i, kxi,  kyi, kzi, exi, eyi, ezi, length, amp, status, ray_index]
            M_final.append(M)

        if M[3] != 0:
            T = -M[0] / M[3]  #  t=-x0/kx (x=0)
            xx0i, yy0i, zz0i = coordinates(M[0], M[1], M[2], M[3], M[4], M[5], T)
            L2 = raydistance(M[0], M[1], M[2], xx0i, yy0i, zz0i)
            Mx = xx0i, yy0i, zz0i, M[3], M[4], M[5], M[6], M[7], M[8], M[9]+L2, amp, status, ray_index
            Ro.append(Mx) # ray at focal point

    with lock:
        result_list.append((M_final, Ro))


def calcRayIntersect(Ri, M1, show_plot:bool = True):
    dt_object = current_time()
    if show_plot:
        print("RayTracing of M1 Start Time =", dt_object)

    M1x = M1[0]
    M1y = M1[1]
    M1z = M1[2]
    SplineParam = spline_calculation.SplineCalculation_M1(M1x, M1y, M1z)

    Rint = []
    Rout = []
    manager = mp.Manager()
    result_list = manager.list()

    num_of_proc = mp.cpu_count()
    chunks = list(chunkify(Ri, len(Ri) // num_of_proc))
    print("num_of_chunks:", len(chunks))
    pool = mp.Pool(processes=num_of_proc)
    print("num_of_proc:", num_of_proc)
    lock = manager.Lock()  # Create a Lock using Manager
    pool.starmap_async(RT, [(chunk, SplineParam, result_list, lock) for chunk in chunks])

    pool.close()
    pool.join()

    for M_final, Ro in result_list:
        Rint.extend(M_final)
        Rout.extend(Ro)

    dt_object_end = current_time()
    print("RayTracing of M1 End Time =", dt_object_end)
    print("RayTracing of M1 Duration =", dt_object_end - dt_object)
    if show_plot:
        plt.show()

    return np.array(Rint), np.array(Rout)


def chunkify(lst, chunk_size):
    # Split the list into smaller chunks
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


if __name__ == '__main__':
    calcRayIntersect()
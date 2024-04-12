from PhysicalScripts.spline import *
from PhysicalScripts.get_mesh import *
from datetime import datetime


def SplineCalculation_M1(M1x, M1y, M1z):
    SplineParam =[]
    [m, x, n, y, z, zl, zr, zu, zd, ibx, iby, MLm] = get_mesh1(M1x, M1y, M1z)
    for q1 in range(m - 1):
        for q2 in range(n - 1):
            xx = (x[q1] + x[q1 + 1]) / 2
            yy = (y[q2] + y[q2 + 1]) / 2
            aaa = (q1 * n + q2) * 16
            MLm[aaa:aaa + 16] = spline_2d(m, x, n, y, z, zl, zr, zu, zd, 0, ibx, iby, xx, yy)

    SplineParam.append(x)
    SplineParam.append(y)
    SplineParam.append(MLm)

    return SplineParam

def SplineCalculation_M2(M2x, M2y, M2z):
    SplineParam = []
    [m, x, n, y, z, zl, zr, zu, zd, ibx, iby, MLm] = get_mesh2(M2x, M2y, M2z)
    for q1 in range(m - 1):
        for q2 in range(n - 1):
            xx = (x[q1] + x[q1 + 1]) / 2
            yy = (y[q2] + y[q2 + 1]) / 2
            aaa = (q1 * n + q2) * 16
            MLm[aaa:aaa + 16] = spline_2d(m, x, n, y, z, zl, zr, zu, zd, 0, ibx, iby, xx, yy)

    SplineParam.append(x)
    SplineParam.append(y)
    SplineParam.append(MLm)

    return SplineParam


def main():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = datetime.fromtimestamp(timestamp)

    print("Start Spline1=", dt_object)
    SplineCalculation_M1()

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object_end = datetime.fromtimestamp(timestamp)
    print("Finish Spline1=", dt_object_end)

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object_end = datetime.fromtimestamp(timestamp)
    print("Finish Spline4=", dt_object_end)


if __name__ == '__main__':
    main()
import numpy as np


def spline_2d(m, x, n, y, z, zl, zr, zu, zd, ind, ibx, iby, xx, yy):
    mm = max(m, n)

    zx = np.zeros(m * n)
    zy = np.zeros(m * n)
    zxy = np.zeros(m * n)
    zz = np.zeros(m * n)
    a = np.zeros(mm)
    b = np.zeros(mm)
    c = np.zeros(mm)
    d = np.zeros(mm)

    su = np.zeros(mm + 1)
    sd = np.zeros(mm + 1)

    if ind == 0:
        for j in range(n):
            aaa = j * m

            zx[aaa:aaa + m], sp, dsp, d2sp = spline_1d(m, x, z[aaa:aaa + m], a, b, c, d, zx[aaa:aaa + m], ind, ibx, zl[j],
                                                    zr[j], xx)

        if iby <= 2:
            sd, sp, dsp, d2sp = spline_1d(m, x, zd, a, b, c, d, sd, 0, ibx, zxy[0], zxy[1], xx)
            su, sp, dsp, d2sp = spline_1d(m, x, zu, a, b, c, d, su, 0, ibx, zxy[2], zxy[3], xx)

        for i in range(m):
            for j in range(n):
                zz[j] = zx[i + j * m]
            aaa = i * n

            zxy[aaa:aaa + n], sp, dsp, d2sp = spline_1d(n, y, zz, a, b, c, d, zxy[aaa:aaa + n], 0, iby, sd[i], su[i], xx)

        for i in range(m):
            for j in range(n):
                zz[j] = z[i + j * m]
            aaa = i * n
            zy[aaa:aaa + n], sp, dsp, d2sp = spline_1d(n, y, zz, a, b, c, d, zy[aaa:aaa + n], 0, iby, zd[i], zu[i], xx)

    for i in range(1, m):
        ni = i
        if x[i] > xx:
            break

    i = ni - 1

    for j in range(1, n):
        nj = j
        if y[j] > yy:
            break

    j = nj - 1

    ml = [z[i + j * m], z[i + 1 + j * m], zx[i + j * m], zx[i + 1 + j * m], z[i + (j + 1) * m], z[i + 1 + (j + 1) * m],
          zx[i + (j + 1) * m], zx[i + 1 + (j + 1) * m], zy[j + i * n], zy[j + (i + 1) * n], zxy[j + i * n],
          zxy[j + (i + 1) * n], zy[j + 1 + i * n], zy[j + 1 + (i + 1) * n], zxy[j + 1 + i * n],
          zxy[j + 1 + (i + 1) * n]]
    return ml


# End of spline_2D
def spline_1d(n, x, y, a, b, c, d, z, ind, ib, ax, bx, xx):
    ne = 0

    if ind == 0:
        a[0] = 2
        ne = n
        ns = 2
        nf = n - 1
        if ib == 1:
            b[0] = 0
            c[0] = 0
            d[0] = 2 * ax
            a[n - 1] = 2
            b[n - 1] = 0
            c[n - 1] = 0
            d[n - 1] = 2 * bx

        if ib == 2:
            b[0] = 1
            c[0] = 0
            h1 = x[1] - x[0]
            d[0] = 3 * (y[1] - y[0]) / h1 - 0.5 * h1 * ax
            a[n - 1] = 2
            b[n - 1] = 0
            c[n - 1] = 1
            h1 = x[n - 1] - x[n - 2]
            d[n - 1] = 3 * (y[n - 1] - y[n - 2]) / h1 + 0.5 * h1 * bx

        if ib == 3:
            h1 = x[1] - x[0]
            h2 = x[n - 1] - x[n - 2]
            am = h2 / (h1 + h2)
            al = 1 - am
            b[0] = am
            c[0] = al
            d[0] = 3 * (am * (y[1] - y[0]) / h1 + al * (y[0] - y[n - 2]) / h2)
            h1 = x[n - 2] - x[n - 3]
            h2 = x[n - 1] - x[n - 2]
            am = h1 / (h1 + h2)
            al = 1 - am
            a[n - 2] = 2
            b[n - 2] = am
            c[n - 2] = al
            d[n - 2] = 3 * (am * (y[n - 1] - y[n - 2]) / h2 + al * (y[n - 2] - y[n - 3]) / h1)
            nf = n - 3
            ne = n - 2

        if ib == 4:
            h1 = x[1] - x[0]
            h2 = x[2] - x[1]
            g0 = h1 / h2
            a[1] = 1 + g0
            b[1] = g0
            c[1] = 0
            am = h1 / (h1 + h2)
            al = 1 - am
            cc = am * (y[2] - y[1]) / h2 + al * (y[1] - y[0]) / h1
            d[1] = cc + 2 * g0 * (y[2] - y[1]) / h2
            h2 = x[n - 1] - x[n - 2]
            h1 = x[n - 2] - x[n - 3]
            gn = h1 / h2
            a[n - 2] = 1 + gn
            b[n - 2] = 0
            c[n - 2] = gn
            am = h1 / (h1 + h2)
            al = 1 - am
            cc = am * (y[n - 1] - y[n - 2]) / h2 + al * (y[n - 2] - y[n - 3]) / h1
            d[n - 2] = cc + 2 * gn * (y[n - 2] - y[n - 3]) / h1
            ns = 3
            nf = n - 3
            ne = n - 3
        # end select

    for j in range(ns - 1, nf):
        h1 = x[j + 1] - x[j]
        h2 = x[j] - x[j - 1]
        am = h2 / (h2 + h1)
        al = 1 - am
        c[j] = al
        a[j] = 2
        b[j] = am
        d[j] = 3 * (am * (y[j + 1] - y[j]) / h1 + al * (y[j] - y[j - 1]) / h2)

    u = np.zeros(ne + 1)
    v = np.zeros(ne + 1)
    w = np.zeros(ne + 1)
    s = np.zeros(ne)
    t = np.zeros(ne)

    u[0] = 0
    v[0] = 0
    w[0] = 1

    for i in range(ne):
        i1 = i + 1
        zp = 1 / (a[i] + c[i] * v[i])
        v[i1] = -b[i] * zp
        u[i1] = (-c[i] * u[i] + d[i]) * zp
        w[i1] = - c[i] * w[i] * zp

    s[ne - 1] = 1
    t[ne - 1] = 0

    for i in range(ne - 2, -1, -1):
        s[i] = v[i + 1] * s[i + 1] + w[i + 1]
        t[i] = v[i + 1] * t[i + 1] + u[i + 1]

    z[ne - 1] = (d[ne - 1] - b[ne - 1] * t[0] - c[ne - 1] * t[ne - 2]) / (
                a[ne - 1] + b[ne - 1] * s[0] + c[ne - 1] * s[ne - 2])

    for i in range(ne - 1):
        z[i] = s[i] * z[ne - 1] + t[i]
        # end of progon

    if ib == 3:
        z[n] = z[1]

    if ib == 4:
        z[1] = g0 ** 2 * z[3] + (g0 ** 2 - 1) * z[2] + 2 * ((y[2] - y[1]) / (x[2] - x[1]) - g0 ** 2 * (y[3] - y[2]) / (x[3] - x[2]))

        z[n] = gn ** 2 * z[n - 2] + [gn ** 2 - 1] * z[n - 1] + 2 * ((y[n] - y[n - 1]) / (x[n] - x[n - 1]) - gn ** 2 * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))
    # end select

    for j in range(1, n):
        nj = j
        if [x[j] > xx]:
            break

    j = nj - 1
    h = x[j + 1] - x[j]
    tt = (xx - x[j]) / h
    rp = (y[j + 1] - y[j]) / h
    aa = -2 * rp + z[j] + z[j + 1]
    bb = -aa + rp - z[j]
    sp = y[j] + (xx - x[j]) * (z[j] + tt * (bb + tt * aa))
    dsp = z[j] + tt * (bb + aa * tt) + tt * (bb + 2 * aa * tt)
    d2sp = (2 * bb + 6 * aa * tt) / h
    return z, sp, dsp, d2sp


# End of spline
def ourspline(xx, yy, x, y, m, n, MLm, ind, status): # ind 1 return only spl
    if (xx < x[0])or (xx > x[m - 1]) or (yy < y[0])or (yy > y[n - 1]):
        status = 1
        return 0, 0, 0, 0,status

    for i in range(1, m):
        ni = i
        if x[i] > xx:
            break

    i = ni - 1

    for j in range(1, n):
        nj = j
        if y[j] > yy:
            break

    j = nj - 1

    aaa = (j * m + i) * 16
    Mu1 = MLm[aaa:aaa + 16]
    Mu = np.reshape(Mu1,(4,4))
    Mu = np.transpose(Mu)
    hx = x[i + 1] - x[i]
    tx = (xx - x[i]) / hx

    hy = y[j + 1] - y[j]
    ty = (yy - y[j]) / hy

    f = [(1 - tx) **2 *(1 + 2 *tx),
         tx **2 *(3 - 2 *tx),
         tx *(1 - tx) **2 *hx,
         -tx **2 *(1 - tx) *hx]
    f = np.reshape(f, (4,1))

    g = [(1 - ty) ** 2 * (1 + 2 * ty), ty ** 2 * (3 - 2 * ty), ty * (1 - ty) ** 2 * hy, - ty ** 2 * (1 - ty) * hy]
    g = np.reshape(g, (1, 4))

    spl = Mu.dot(f)
    spl = g.dot(spl)

    if ind == 1:
        return spl, 0, 0, 0, status


    dfdx = [-6 * tx * (1 - tx),
            2 * tx * (3 - 2 * tx) - 2 * tx ** 2,
            ((1 - tx) ** 2 - tx * 2 * (1 - tx)) * hx,
            (-2 * tx * (1 - tx) + tx ** 2) * hx] / hx
    dfdx = np.reshape(dfdx, (4,1))

    dgdy = [-6 * ty * (1 - ty), 2 * ty * (3 - 2 * ty) - 2 * ty ** 2, ((1 - ty) ** 2 - ty * 2 * (1 - ty)) * hy, (-2 * ty * (1 - ty) + ty ** 2) * hy] / hy
    dgdy= np.reshape(dgdy, (1, 4))
    spl = Mu.dot(f)
    spl = g.dot(spl)
    Nx = Mu.dot(dfdx)
    Nx = g.dot(Nx)
    Ny = Mu.dot(f)
    Ny = dgdy.dot(Ny)
    Nz = 1

    Norm = np.sqrt(Nx**2 +Ny**2 +Nz**2)

    Nx = -Nx / Norm
    Ny = -Ny / Norm
    Nz = Nz / Norm

    return spl, Nx, Ny, Nz, status

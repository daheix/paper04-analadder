#!/usr/bin/env python3
"""fem_check.py — 独立 P1 节点型 Az FEM, 与 GetDP Form1P 交叉验证.

用法: python3 fem_check.py <mesh.msh> [--mur-iron 7000] [--mur-shaft 1]
读 msh2 网格, 物理组: 1=shaft 2=rotor_iron 3=magnet_N 4=magnet_S 5=airgap
6,8-12=槽(空气) 7=定子铁心. 输出气隙 r=29.5mm 圆周 Bn 剖面与极下均值.
"""
import sys
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4e-7 * math.pi
BR, MUR_MAG, HC = 1.16, 1.05, 1.16 / (1.05 * MU0)
R_SO, P, ALPHA = 0.050, 2, 0.75


def read_msh2(path):
    nodes = {}
    tris = []  # (n1,n2,n3, phys)
    sec = None
    for ln in open(path):
        t = ln.split()
        if not t:
            continue
        if t[0].startswith('$'):
            sec = t[0][1:]
            continue
        if sec == 'Nodes':
            try:
                nid = int(t[0])
                if len(t) >= 3:
                    nodes[nid] = (float(t[1]), float(t[2]))
            except ValueError:
                pass
        elif sec == 'Elements':
            if len(t) < 3:
                continue
            try:
                eid, etype, ntags = int(t[0]), int(t[1]), int(t[2])
            except ValueError:
                continue
            if etype == 2:
                phys = int(t[3])
                ns = [int(v) for v in t[3 + ntags:3 + ntags + 3]]
                tris.append((ns[0], ns[1], ns[2], phys))
    return nodes, tris


def main():
    mesh = sys.argv[1]
    mur_iron = float(sys.argv[sys.argv.index('--mur-iron') + 1]) if '--mur-iron' in sys.argv else 7000.0
    mur_shaft = float(sys.argv[sys.argv.index('--mur-shaft') + 1]) if '--mur-shaft' in sys.argv else 1.0

    nodes, tris = read_msh2(mesh)
    ids = sorted(nodes)
    idx = {nid: k for k, nid in enumerate(ids)}
    XY = np.array([nodes[i] for i in ids])
    N = len(ids)
    nu = {1: 1 / (MU0 * mur_shaft), 2: 1 / (MU0 * mur_iron),
          3: 1 / (MU0 * MUR_MAG), 4: 1 / (MU0 * MUR_MAG), 5: 1 / MU0,
          7: 1 / (MU0 * mur_iron)}
    for t6 in (6, 8, 9, 10, 11, 12):  # 槽=空气; 7=定子铁心!
        nu[t6] = 1 / MU0

    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    Bx = np.zeros(len(tris)); By = np.zeros(len(tris))
    for m, (a, b, c, phys) in enumerate(tris):
        ia, ib, ic = idx[a], idx[b], idx[c]
        xa, ya = XY[ia]; xb, yb = XY[ib]; xc, yc = XY[ic]
        twoA = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
        assert abs(twoA) > 1e-18, f"degenerate tri {a},{b},{c}"
        # 梯度: grad N_i = (b_i, c_i)/(2A)
        b1, c1 = yb - yc, xc - xb
        b2, c2 = yc - ya, xa - xc
        b3, c3 = ya - yb, xb - xa
        nloc = [ia, ib, ic]
        bl = [b1, b2, b3]; cl = [c1, c2, c3]
        k = nu[phys]
        for i in range(3):
            for j in range(3):
                rows.append(nloc[i]); cols.append(nloc[j])
                vals.append(k * (bl[i] * bl[j] + cl[i] * cl[j]) / (2 * abs(twoA)))
        # PM 源: f_i = int Hc . grad N_i dA = ±(Hc_x b_i - Hc_y c_i)/2
        if phys in (3, 4):
            th = math.atan2((ya + yb + yc) / 3, (xa + xb + xc) / 3)
            s = 1.0 if phys == 3 else -1.0
            hx, hy = s * HC * math.cos(th), s * HC * math.sin(th)
            for i in range(3):
                rhs[nloc[i]] += 0.5 * (hx * cl[i] - hy * bl[i]) * (1 if twoA > 0 else -1)

    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    # Dirichlet: r ≈ R_so 外边界
    r = np.hypot(XY[:, 0], XY[:, 1])
    fixed = np.where(r > R_SO - 1e-6)[0]
    free = np.setdiff1d(np.arange(N), fixed)
    rhs -= A[:, fixed] @ np.zeros(len(fixed))  # A_fixed = 0
    Af = A[free][:, free]
    sol = np.zeros(N)
    sol[free] = spsolve(Af.tocsc(), rhs[free])
    print(f"N={N}, tris={len(tris)}, fixed={len(fixed)}, solve done")

    # 单元 B
    Bxe = np.zeros(len(tris)); Bye = np.zeros(len(tris))
    for m, (a, b, c, phys) in enumerate(tris):
        ia, ib, ic = idx[a], idx[b], idx[c]
        xa, ya = XY[ia]; xb, yb = XY[ib]; xc, yc = XY[ic]
        twoA = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
        b1, c1 = yb - yc, xc - xb
        b2, c2 = yc - ya, xa - xc
        b3, c3 = ya - yb, xb - xa
        av = [sol[ia], sol[ib], sol[ic]]
        # Bx = dA/dy = sum A_i c_i / twoA ; By = -dA/dx = -sum A_i b_i / twoA
        Bxe[m] = (c1 * av[0] + c2 * av[1] + c3 * av[2]) / twoA
        Bye[m] = -(b1 * av[0] + b2 * av[1] + b3 * av[2]) / twoA

    Xc = (XY[[idx[t[0]] for t in tris], 0] + XY[[idx[t[1]] for t in tris], 0] + XY[[idx[t[2]] for t in tris], 0]) / 3
    Yc = (XY[[idx[t[0]] for t in tris], 1] + XY[[idx[t[1]] for t in tris], 1] + XY[[idx[t[2]] for t in tris], 1]) / 3
    r_e = np.hypot(Xc, Yc)
    gap = (r_e > 0.0291) & (r_e < 0.0299)
    th_e = np.degrees(np.arctan2(Yc, Xc)) % 360
    Bn = Bxe * np.cos(np.radians(th_e)) + Bye * np.sin(np.radians(th_e))
    prof = np.zeros(360); cnt = np.zeros(360)
    for b_, v in zip((th_e[gap] // 1).astype(int), Bn[gap]):
        prof[int(b_) % 360] += v; cnt[int(b_) % 360] += 1
    m2 = cnt > 0; prof[m2] /= cnt[m2]
    np.save('/tmp/fem_profile.npy', prof)
    pole = np.abs(np.mod(np.arange(360) - 0, 360) - 180) < 33.75
    print(f"FEM 极下(0±33.75°) <Bn> = {prof[pole].mean():+.4f} T")
    print("每15°: " + " ".join(f"{prof[i]:+.3f}" for i in range(0, 360, 15)))


if __name__ == '__main__':
    main()

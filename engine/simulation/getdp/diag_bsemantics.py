#!/usr/bin/env python3
"""诊断: b_map.pos 9 值语义 (node1 vs 3节点均值) 对 Maxwell 应力转矩的影响."""
import math
import numpy as np

from run_emag_getdp import parse_pos_vv, P, MU0
from fem_check import read_msh2

nodes, tris = read_msh2('motor_sector.msh')
gap = []
for (a, b, c, phys) in tris:
    if phys != 5:
        continue
    p = [nodes[a], nodes[b], nodes[c]]
    r = [math.hypot(x, y) for x, y in p]
    if P['R_rm'] + 1e-4 < sum(r) / 3 < P['R_si'] - 1e-4:
        gap.append(p)
gap = np.asarray(gap)
cen = gap.mean(axis=1)


def torque(f, r_s, mode, n=1440):
    cs, vs = parse_pos_vv(f)
    cx = cs[:, 0::3].mean(axis=1)
    cy = cs[:, 1::3].mean(axis=1)
    if mode == 'node1':
        bx_all, by_all = vs[:, 0], vs[:, 1]
    else:
        bx_all = vs[:, 0::3].mean(axis=1)
        by_all = vs[:, 1::3].mean(axis=1)
    bmap = {(round(x, 6), round(y, 6)): (bx, by)
            for x, y, bx, by in zip(cx, cy, bx_all, by_all)}
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    px, py = r_s * np.cos(th), r_s * np.sin(th)
    p0, p1, p2 = gap[:, 0], gap[:, 1], gap[:, 2]
    dd = ((p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0])
          + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1]))
    X, Y = px[:, None] - p2[None, :, 0], py[:, None] - p2[None, :, 1]
    l1 = ((p1[:, 1] - p2[:, 1])[None, :] * X + (p2[:, 0] - p1[:, 0])[None, :] * Y) / dd[None, :]
    l2 = ((p2[:, 1] - p0[:, 1])[None, :] * X + (p0[:, 0] - p2[:, 0])[None, :] * Y) / dd[None, :]
    l3 = 1 - l1 - l2
    hit = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
    j = np.argmax(hit, axis=1)
    ok = hit[np.arange(n), j]
    if not ok.all():
        d2 = (px[:, None] - cen[None, :, 0]) ** 2 + (py[:, None] - cen[None, :, 1]) ** 2
        j[~ok] = np.argmin(d2[~ok], axis=1)
    bx, by = bmap[(round(cen[j, 0][0], 6), round(cen[j, 0][1], 6))][0] * 0, 0  # noqa
    # 向量化取值
    bx = np.array([bmap[(round(cen[k, 0], 6), round(cen[k, 1], 6))][0] for k in j])
    by = np.array([bmap[(round(cen[k, 0], 6), round(cen[k, 1], 6))][1] for k in j])
    ct, st = np.cos(th), np.sin(th)
    Bn, Bt = bx * ct + by * st, -bx * st + by * ct
    return P['L_stack'] * r_s ** 2 * (Bn * Bt / MU0).mean() * 2 * math.pi


if __name__ == '__main__':
    for mode in ('node1', 'mean3'):
        t90 = torque('b_map_dl_90b.pos', 0.0295, mode)
        tA = torque('b_map_dl2.pos', 0.0295, mode)
        tnl = torque('b_map_nl2.pos', 0.0295, mode)
        print(f'{mode:6s}: T(平衡90)={t90:+.4f}  T(仅A)={tA:+.4f}  T(无载)={tnl:+.4f}')

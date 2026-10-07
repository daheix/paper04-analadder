#!/usr/bin/env python3
"""virtual_work.py — 虚功法转矩 (金标准) vs GetDP Maxwell 应力对账.

原理: 转子为同心圆环 (轴对称磁路), 磁极图形旋转 α 只需把 hc 按角度重指派,
网格/磁导率分布不变 → 同一网格多 α 求解严格精确。
W'(α) = ½Σν|B|²·A·L (线性介质共能), T = +∂W'/∂α (固定电流)。

用法: python3 virtual_work.py <mesh.msh>
对比: GetDP Maxwell T(A-only js=2e6)=−0.4430, T(平衡δ90)=−0.9322, 空载齿槽+0.1449
"""
import sys, math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4e-7 * math.pi
MUR_MAG, HC = 1.05, 1.16 / (1.05 * MU0)
R_SO, LSTK = 0.050, 0.1
MUR_IRON, MUR_SHAFT = 7000.0, 1.0
A_SLOT = 3.016e-5  # 每槽导体区面积 m²


def read_msh2(path):
    nodes, tris = {}, []
    sec = None
    for ln in open(path):
        t = ln.split()
        if not t:
            continue
        if t[0].startswith('$'):
            sec = t[0][1:]
            continue
        if sec == 'Nodes' and len(t) >= 3:
            try:
                nodes[int(t[0])] = (float(t[1]), float(t[2]))
            except ValueError:
                pass
        elif sec == 'Elements' and len(t) >= 3:
            try:
                etype, ntags = int(t[1]), int(t[2])
            except ValueError:
                continue
            if etype == 2:
                tris.append(tuple(int(v) for v in t[3 + ntags:6 + ntags]) +
                            (int(t[3]),))
    return nodes, tris


class FemModel:
    def __init__(self, mesh):
        nodes, tris = read_msh2(mesh)
        ids = sorted(nodes)
        self.idx = {n: k for k, n in enumerate(ids)}
        self.XY = np.array([nodes[i] for i in ids])
        self.N = len(ids)
        self.tris = tris
        nu_map = {1: 1 / (MU0 * MUR_SHAFT), 2: 1 / (MU0 * MUR_IRON),
                  3: 1 / (MU0 * MUR_MAG), 4: 1 / (MU0 * MUR_MAG),
                  5: 1 / MU0, 7: 1 / (MU0 * MUR_IRON)}
        for t in (6, 8, 9, 10, 11, 12):
            nu_map[t] = 1 / MU0
        self.nu_e = np.array([nu_map[t[3]] for t in tris])
        # 单元几何 (梯度/面积) 预计算
        E = len(tris)
        self.bl = np.zeros((E, 3)); self.cl = np.zeros((E, 3))
        self.twoA = np.zeros(E)
        self.nloc = np.zeros((E, 3), dtype=int)
        for m, (a, b, c, phys) in enumerate(tris):
            ia, ib, ic = self.idx[a], self.idx[b], self.idx[c]
            xa, ya = self.XY[ia]; xb, yb = self.XY[ib]; xc, yc = self.XY[ic]
            ta = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
            self.twoA[m] = ta
            self.bl[m] = [yb - yc, yc - ya, ya - yb]
            self.cl[m] = [xc - xb, xa - xc, xb - xa]
            self.nloc[m] = [ia, ib, ic]
        self.area = np.abs(self.twoA) / 2
        # 组装刚度阵 (ν 固定): k_ij = ν(b_i b_j + c_i c_j)/(2A)
        i3 = np.repeat(self.nloc, 3, axis=1).ravel()
        j3 = np.tile(self.nloc, (1, 3)).ravel()
        BB = self.bl[:, :, None] * self.bl[:, None, :] + \
             self.cl[:, :, None] * self.cl[:, None, :]      # (E,3,3)
        k3 = (self.nu_e[:, None, None] * BB /
              (2 * np.abs(self.twoA))[:, None, None]).ravel()
        self.K = coo_matrix((k3, (i3, j3)), shape=(self.N, self.N)).tocsr()
        r = np.hypot(self.XY[:, 0], self.XY[:, 1])
        self.fixed = np.where(r > R_SO - 1e-6)[0]
        self.free = np.setdiff1d(np.arange(self.N), self.fixed)
        self.Kff = self.K[self.free][:, self.free].tocsc()
        # 磁体单元掩码 + 质心角 + 顶点角度跨度 (供旋转窗分数磁化)
        mag = np.array([t[3] in (3, 4) for t in tris])
        self.mag_m = np.where(mag)[0]
        xc = self.XY[self.nloc[:, 0], 0] + self.XY[self.nloc[:, 1], 0] + self.XY[self.nloc[:, 2], 0]
        yc = self.XY[self.nloc[:, 0], 1] + self.XY[self.nloc[:, 1], 1] + self.XY[self.nloc[:, 2], 1]
        self.th_e = np.arctan2(yc / 3, xc / 3)
        thv = np.arctan2(self.XY[self.nloc[mag]][:, :, 1],
                         self.XY[self.nloc[mag]][:, :, 0])          # (M,3)
        self.mag_thv = thv - self.th_e[mag][:, None]              # 相对质心
        self.mag_w = self.mag_thv.max(1) - self.mag_thv.min(1)    # 角宽 [rad]
        # 槽电流: 标签→(角度°, 电流密度) 由外部传入
        self.slot_tag = {6: 0, 8: 30, 9: 60, 10: 90, 11: 120, 12: 150}  # Ap,Cn,Bp,An,Cp,Bn @°
        self.slot_m = {tag: np.array([m for m, t in enumerate(tris) if t[3] == tag])
                       for tag in self.slot_tag}

    def rhs_pm(self, alpha):
        """磁极图形旋转 alpha [rad]: N 窗 (半宽 33.75°=0.75·极距/2) 随 α 平移,
        边界单元按角度重叠比例分数磁化 s=2·frac−1 → W'(α) 光滑。
        N 极中心 @α+k·90°, 磁化方向恒为径向 r̂."""
        m = self.mag_m
        th = self.th_e[m]
        # 质心相对最近 N 中心 (α+k·180°) 的偏差, wrap 到 (−90°,90°]
        P = math.pi
        d = th - alpha - np.round((th - alpha) / P) * P
        # N 窗边界距质心 ±h; 单元角宽 w: 覆盖比例 (磁体域边界与网格共形,
        # α=0 时无单元跨界 → 严格复现 GetDP 磁体布局)
        h = math.radians(33.75)
        lo, hi = d - self.mag_w / 2, d + self.mag_w / 2
        ov = np.clip(np.minimum(hi, h) - np.maximum(lo, -h), 0.0, self.mag_w)
        frac = ov / self.mag_w
        s = 2.0 * frac - 1.0
        hx = s * HC * np.cos(th)
        hy = s * HC * np.sin(th)
        bl, cl = self.bl[m], self.cl[m]
        f = 0.5 * (hx[:, None] * cl - hy[:, None] * bl)
        sgn = np.sign(self.twoA[m])
        rhs_add = np.zeros(self.N)
        for k in range(3):
            np.add.at(rhs_add, self.nloc[m, k], f[:, k] * sgn)
        return rhs_add

    def rhs_js(self, js_by_tag):
        """槽电流源: f_i = ∫ Jz N_i dA = Jz·Area/3."""
        rhs = np.zeros(self.N)
        for tag, js in js_by_tag.items():
            if js == 0:
                continue
            m = self.slot_m[tag]
            np.add.at(rhs, self.nloc[m].ravel(),
                      np.repeat(js * self.area[m] / 3, 3))
        return rhs

    def solve(self, rhs):
        b = rhs[self.free]
        sol = np.zeros(self.N)
        sol[self.free] = spsolve(self.Kff, b)
        return sol

    def field_B(self, sol):
        av = sol[self.nloc]
        Bx = (self.cl * av).sum(1) / self.twoA
        By = -(self.bl * av).sum(1) / self.twoA
        return Bx, By

    def coenergy(self, Bx, By):
        return float(0.5 * (self.nu_e * (Bx**2 + By**2) * self.area).sum() * LSTK)


def torque_maxwell(drv, pos_path, r=0.0295):
    return drv.torque(pos_path, r_sample=r)


def main():
    mesh = sys.argv[1] if len(sys.argv) > 1 else "motor_sector.msh"
    fm = FemModel(mesh)
    print(f"mesh: N={fm.N}, tris={len(fm.tris)}, fixed={len(fm.fixed)}")
    # 网格标签体检
    tags = {}
    for t in fm.tris:
        tags[t[3]] = tags.get(t[3], 0) + 1
    print(f"物理标签: {dict(sorted(tags.items()))}")
    for tag, m in fm.slot_m.items():
        if len(m):
            ths = np.degrees(fm.th_e[m]) % 360
            print(f"  tag{tag}: n={len(m)} 角度样本 {sorted(set((ths//10*10).astype(int)))[:8]}")
    # 无载验证 @α=0: 极均值 Bn vs GetDP 0.7585
    rhs0 = fm.rhs_pm(0.0)
    print(f"rhs_pm 非零项: {np.count_nonzero(rhs0)}, rhs 范围 [{rhs0.min():.3e},{rhs0.max():.3e}]")
    sol = fm.solve(rhs0)
    print(f"A 范围 [{sol.min():.3e}, {sol.max():.3e}]")
    Bx, By = fm.field_B(sol)
    print(f"|B| max={np.hypot(Bx,By).max():.4f} T")
    r_e = np.hypot(*(fm.XY[fm.nloc].mean(1)).T)
    gap = (r_e > 0.0291) & (r_e < 0.0299)
    thg = np.degrees(fm.th_e[gap]) % 360
    Bng = Bx[gap] * np.cos(np.radians(thg)) + By[gap] * np.sin(np.radians(thg))
    Btg = -Bx[gap] * np.sin(np.radians(thg)) + By[gap] * np.cos(np.radians(thg))
    print(f"气隙: |Bn|max={np.abs(Bng).max():.4f} |Bt|max={np.abs(Btg).max():.4f}")
    print(f"无载校验: 极下(0°±33.75°) <Bn> = {Bng[(thg<33.75)|(thg>326.25)].mean():+.4f} T (GetDP: +0.7585)")
    print(f"          气隙 Bn 每15°: " + " ".join(f"{int(t)}°:{v:+.3f}" for t, v in
          zip(np.arange(0, 360, 15), [Bng[(np.abs(((thg - a + 180) % 360) - 180)) < 7.5].mean() for a in np.arange(0, 360, 15)])))

    def vw_curve(label, js_by_tag, alphas_deg):
        """T = +dW'/dα 中心差分."""
        Ts = []
        for adeg in alphas_deg:
            a = math.radians(adeg)
            rhs = fm.rhs_pm(a) + fm.rhs_js(js_by_tag)
            sol = fm.solve(rhs)
            Bx, By = fm.field_B(sol)
            Ts.append(fm.coenergy(Bx, By))
        alphas = np.radians(np.array(alphas_deg))
        dW = np.gradient(np.array(Ts), alphas)  # J/rad
        print(f"[{label}] W'(α) = {[f'{w:.6f}' for w in Ts]} J")
        for adeg, t in zip(alphas_deg, dW):
            print(f"    α={adeg:+.2f}°: T_vw={t:+.4f} Nm")
        return dW

    # ① 空载齿槽转矩
    vw_curve("空载 (i=0)", {}, [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    # ② 仅 A 相 (Ap=+js, An=−js) — GetDP Maxwell: −0.4430
    vw_curve("仅A相 js=2e6", {6: 2e6, 10: -2e6}, [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    # ③ 平衡 δ=90° — GetDP Maxwell: −0.9322, 正弦拟合幅值 1.080
    vw_curve("平衡δ90 js=2e6", {6: 2e6, 10: -2e6, 9: -1e6, 12: 1e6,
                                11: -1e6, 8: 1e6}, [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])


if __name__ == '__main__':
    main()

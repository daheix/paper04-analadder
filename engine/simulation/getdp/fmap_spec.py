#!/usr/bin/env python3
"""M99 磁链 MAP-ROM 口径卡 (doctest 可执行) — 双线性 λd/λq LUT + 外推披露 + dq 电流方程离散化

链路: fmap_spec.py (本卡) + run_fluxmap_rom.py (采样驱动器)
      + constants/fluxmap_design.json (fmap_* 白名单)
      → 库模型静磁 (id,iq) 网格采样 → fluxmap_rom.json (轴/网格/数值 + 插值误差自检)
      → M97 FMU 端口扩展: flux_map_mode 枚举 (analytic_dq / fluxmap_lut 并列可选, FMI v2)

口径 (LUT 提取 — 静磁工作点, 全披露):
    转子 d 轴对准 A+ 槽 (θe=0), Park 逆变换 (幅值不变式):
        ia = id,  ib = −id/2 + (√3/2)·iq,  ic = −id/2 − (√3/2)·iq
    槽中部半径 A(z) 环向剖面最小二乘拟合 p 次谐波: f(θ) = A·cos pθ + B·sin pθ
        λd = 4·Nc·L·A,  λq = 4·Nc·L·B   (与 flux_linkage λm=4NcL·max|f| 同族口径)
    转矩一致性: Te = (3/2)·p·(λd·iq − λq·id)  vs Maxwell 应力 FEM 转矩
    (槽带单带采样 vs 能量法 ~14% 系统偏差已披露, 容限 25%)

口径 (双线性插值 + 网格外推披露规则):
    网格内: 双线性 (t,s ∈ [0,1]); 网格外: 同式 t/s 线性延伸 (各轴末端梯度),
    extrap_flag 披露标记 → 报告披露外推评估计数 (网格外饱和物理未采样)。

口径 (dq 电流方程离散化 — 增量电感 LUT 形式):
    λd(id,iq), λq(id,iq) 双线性 LUT; 增量 (切线) 电感 = LUT 中心差分:
        Ld_inc = ∂λd/∂id, Lq_inc = ∂λq/∂iq   (步长 fmap_did_h_A / fmap_diq_h_A)
        did/dt = (ud − Rs·id + ωe·λq) / Ld_inc
        diq/dt = (uq − Rs·iq − ωe·λd) / Lq_inc
    显式 Euler 定步长; 表观磁链状态形式 (λ 为状态量) 未采用, 差异披露。

>>> spec = load_fmap()
>>> spec["fmap_extrap_rule"], spec["fmap_interp_tol_pct"]
('linear_edge_disclosed', 2.0)
>>> # ---- Park 逆变换锚点 (幅值不变式, θe=0, 逐位) ----
>>> idq_to_abc(1.0, 0.0)
(1.0, -0.5, -0.5)
>>> abc = idq_to_abc(0.0, 1.0)
>>> round(abc[0], 12), round(abc[1], 12), round(abc[2], 12)
(0.0, 0.866025403784, -0.866025403784)
>>> # 幅值不变: |ia|²+|ib|²+|ic|² = (3/2)·(id²+iq²)
>>> round(sum(a*a for a in idq_to_abc(1.0, 1.0)), 12), 1.5*(1.0+1.0)
(3.0, 3.0)
>>> # ---- 谐波分解锚点: f=cos2θ+0.5sin2θ → A=1, B=0.5 (整周期最小二乘精确) ----
>>> th = list(range(0, 360, 10))
>>> vals = [math.cos(math.radians(2*t)) + 0.5*math.sin(math.radians(2*t)) for t in th]
>>> a, b = dq_flux_decompose(th, vals, 2)
>>> round(a, 9), round(b, 9)
(1.0, 0.5)
>>> vals_q = [math.sin(math.radians(2*t)) for t in th]
>>> round(dq_flux_decompose(th, vals_q, 2)[0], 9), round(dq_flux_decompose(th, vals_q, 2)[1], 9)
(0.0, 1.0)
>>> # λ 口径: λd = 4·Nc·L·A → Nc=100, L=0.1, A=0.5 → 20.0 Wb (纯算例)
>>> round(lam_from_ab(0.5, 0.25, 100, 0.1)[0], 9), round(lam_from_ab(0.5, 0.25, 100, 0.1)[1], 9)
(20.0, 10.0)
>>> # ---- 双线性插值锚点 (手工算例, 逐位): xs=ys=[0,1], g=[[0,1],[2,3]] ----
>>> xs, ys, g = [0.0, 1.0], [0.0, 1.0], [[0.0, 1.0], [2.0, 3.0]]
>>> bilinear(xs, ys, g, 0.5, 0.5)
1.5
>>> round(bilinear(xs, ys, g, 0.25, 0.75), 12)
1.25
>>> # 网格节点逐位复现 (插值退化为查表)
>>> bilinear(xs, ys, g, 1.0, 0.0)
2.0
>>> # ---- 外推披露: 网格外 = 各轴线性延伸 + 标记 ----
>>> extrap_flag(xs, ys, 0.5, 0.5)
False
>>> extrap_flag(xs, ys, 1.5, 0.5), extrap_flag(xs, ys, 0.5, -0.1)
(True, True)
>>> round(bilinear(xs, ys, g, 1.5, 0.5), 9), round(bilinear(xs, ys, g, -0.5, 0.5), 9)
(3.5, -0.5)
>>> # ---- dq 电流方程锚点: 线性 LUT (λd=0.1+0.002·id, λq=0.003·iq) 退化回解析 ----
>>> xs3, ys3 = [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]
>>> lut = {"id": xs3, "iq": ys3,
...        "lam_d": [[0.1 + 0.002*x for y in ys3] for x in xs3],
...        "lam_q": [[0.003*y for y in ys3] for x in xs3]}
>>> sp = lut_spec_lite()
>>> d_id, d_iq = dq_derivs(1.0, 1.0, 100.0, 0.0, 10.0, lut, sp)
>>> round(d_id, 6), round(d_iq, 6)
(-100.0, -233.333333)
>>> # did/dt = (0−0.5·1+100·0.003)/0.002 = −0.2/0.002 = −100
>>> # diq/dt = (10−0.5·1−100·0.102)/0.003 = −0.7/0.003 = −233.33
>>> # 显式 Euler 一步 (dt=1e-4): id1=0.99, iq1=1−0.0233333=0.976667
>>> s1 = dq_euler_step(1.0, 1.0, 100.0, 0.0, 10.0, lut, sp, 1e-4)
>>> round(s1[0], 6), round(s1[1], 6)
(0.99, 0.976667)
>>> # ---- 转矩锚点: id=0, iq=10/3, λd(0,·)=0.1 → Te=1.5·p·λm·iq = 1.0 Nm ----
>>> round(te_from_lut(0.0, 10.0/3.0, lut, 2), 9)
1.0
>>> # ---- 相对误差口径 (复用 fmu_spec.rel_err, 分母含绝对下限) ----
>>> round(rel_err(1.0001, 1.0, 0.001) * 100.0, 6)
0.01
>>> # ---- 非法输入 fail-fast (禁静默兜底) ----
>>> bilinear([1.0, 0.0], ys, g, 0.5, 0.5)
Traceback (most recent call last):
    ...
ValueError: 轴必须严格递增
>>> dq_derivs(1.0, 1.0, 100.0, 0.0, 10.0, lut,
...           lut_spec_lite(ld_inc=-0.002, lq_inc=0.003))
Traceback (most recent call last):
    ...
ValueError: 增量电感必须为正: Ld_inc=-0.002, Lq_inc=0.003
>>> load_fmap_bad()
Traceback (most recent call last):
    ...
KeyError: 'fluxmap_design.json 缺字段: fmap_id_grid_A'
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import fluxmap_design
from fmu_spec import rel_err  # 复用 M97 相对误差口径 (唯一实现)

_FMAP_REQ_KEYS = [
    "fmap_model_whitelist", "fmap_id_grid_A", "fmap_iq_grid_A",
    "fmap_i_abs_max_A", "fmap_extrap_rule", "fmap_interp_tol_pct",
    "fmap_torque_consist_tol_pct", "fmap_lambda_abs_floor_Wb",
    "fmap_check_points_max", "fmap_did_h_A", "fmap_diq_h_A", "fmap_caveats",
]


def load_fmap():
    """fmap_* 白名单 (fail-fast) → spec dict (缺字段直接 KeyError, 禁静默兜底)。"""
    raw = fluxmap_design()
    for k in _FMAP_REQ_KEYS:
        if k not in raw:
            raise KeyError(f"fluxmap_design.json 缺字段: {k}")
    out = {k: raw[k] for k in _FMAP_REQ_KEYS}
    out["caveats"] = raw["fmap_caveats"]
    return out


def load_fmap_bad():
    """doctest 用: 缺字段白名单装载必须 fail-fast。"""
    raw = dict(fluxmap_design())
    del raw["fmap_id_grid_A"]
    for k in _FMAP_REQ_KEYS:
        if k not in raw:
            raise KeyError(f"fluxmap_design.json 缺字段: {k}")
    return raw


def lut_spec_lite(rs=0.5, p=2, did_h=0.001, diq_h=0.001,
                  ld_inc=None, lq_inc=None):
    """测试/锚点用 LUT spec (ld_inc/lq_inc 覆盖仅供 fail-fast 注入)。"""
    return {"Rs": rs, "p": int(p), "did_h": did_h, "diq_h": diq_h,
            "_ld_inc": ld_inc, "_lq_inc": lq_inc}


def idq_to_abc(id_c, iq):
    """Park 逆变换 (幅值不变式, θe=0, d 轴对准 A+ 槽):
    ia=id, ib=−id/2+(√3/2)·iq, ic=−id/2−(√3/2)·iq。"""
    s3h = 0.5 * math.sqrt(3.0)
    return (id_c, -0.5 * id_c + s3h * iq, -0.5 * id_c - s3h * iq)


def dq_flux_decompose(theta_deg, vals, p):
    """环向剖面 f(θ) 最小二乘拟合 p 次谐波 → (A, B): f ≈ A·cos pθ + B·sin pθ。

    均匀角度采样下 cos pθ / sin pθ 整周期正交 (Σcos·sin = 0), 解析投影:
    A = Σf·cos pθ / Σcos² pθ, B = Σf·sin pθ / Σsin² pθ。
    """
    n = len(theta_deg)
    if n == 0:
        raise ValueError("采样为空")
    cs = [math.cos(math.radians(p * t)) for t in theta_deg]
    sn = [math.sin(math.radians(p * t)) for t in theta_deg]
    scc = sum(c * c for c in cs)
    sss = sum(s * s for s in sn)
    if scc <= 0.0 or sss <= 0.0:
        raise ValueError("谐波基退化 (Σcos²/Σsin² ≤ 0)")
    a = sum(v * c for v, c in zip(vals, cs)) / scc
    b = sum(v * s for v, s in zip(vals, sn)) / sss
    return a, b


def lam_from_ab(a, b, nc, l_stack):
    """谐波系数 → (λd, λq) = (4·Nc·L·A, 4·Nc·L·B) (与 flux_linkage 同族口径)。"""
    return 4.0 * nc * l_stack * a, 4.0 * nc * l_stack * b


def bilinear(xs, ys, grid, x, y):
    """规则网格双线性插值; 网格外 = 各轴线性延伸 (t/s 不截断, 外推披露规则)。

    grid[i][j], i↔xs (id 轴), j↔ys (iq 轴)。
    """
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("轴至少 2 点")
    if any(xs[i + 1] <= xs[i] for i in range(len(xs) - 1)) \
            or any(ys[j + 1] <= ys[j] for j in range(len(ys) - 1)):
        raise ValueError("轴必须严格递增")
    if len(grid) != len(xs) or any(len(row) != len(ys) for row in grid):
        raise ValueError("网格形状与轴不匹配")
    i = max(0, min(len(xs) - 2, _bisect_span(xs, x)))
    j = max(0, min(len(ys) - 2, _bisect_span(ys, y)))
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    s = (y - ys[j]) / (ys[j + 1] - ys[j])
    return ((1 - t) * (1 - s) * grid[i][j] + (1 - t) * s * grid[i][j + 1]
            + t * (1 - s) * grid[i + 1][j] + t * s * grid[i + 1][j + 1])


def _bisect_span(xs, x):
    """x 所跨单元起点 i (xs[i] ≤ x < xs[i+1]); 越界返回 0 或 len−2 (线性延伸)。"""
    lo, hi = 0, len(xs) - 1
    if x < xs[0]:
        return 0
    if x >= xs[hi]:
        return hi - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


def extrap_flag(xs, ys, x, y):
    """网格外推披露标记: (x,y) 落在采样矩形外 → True。"""
    return not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1])


def te_from_lut(id_c, iq, lut, p):
    """LUT 转矩: Te = (3/2)·p·(λd·iq − λq·id) (与 M96/M98 dq 转矩同口径)。"""
    lam_d = bilinear(lut["id"], lut["iq"], lut["lam_d"], id_c, iq)
    lam_q = bilinear(lut["id"], lut["iq"], lut["lam_q"], id_c, iq)
    return 1.5 * p * (lam_d * iq - lam_q * id_c)


def dq_derivs(id_c, iq, we, ud, uq, lut, sp):
    """dq 电流方程右端 (增量电感 LUT 形式) → (did/dt, diq/dt):

    Ld_inc = ∂λd/∂id, Lq_inc = ∂λq/∂iq (LUT 中心差分, 步长 sp["did_h"/"diq_h"]);
    did/dt = (ud − Rs·id + ωe·λq)/Ld_inc;  diq/dt = (uq − Rs·iq − ωe·λd)/Lq_inc。
    """
    h_d, h_q = sp["did_h"], sp["diq_h"]
    if sp.get("_ld_inc") is not None:
        ld_inc = sp["_ld_inc"]
    else:
        ld_inc = (bilinear(lut["id"], lut["iq"], lut["lam_d"], id_c + h_d, iq)
                  - bilinear(lut["id"], lut["iq"], lut["lam_d"], id_c - h_d, iq)
                  ) / (2.0 * h_d)
    if sp.get("_lq_inc") is not None:
        lq_inc = sp["_lq_inc"]
    else:
        lq_inc = (bilinear(lut["id"], lut["iq"], lut["lam_q"], id_c, iq + h_q)
                  - bilinear(lut["id"], lut["iq"], lut["lam_q"], id_c, iq - h_q)
                  ) / (2.0 * h_q)
    if ld_inc <= 0.0 or lq_inc <= 0.0:
        raise ValueError(f"增量电感必须为正: Ld_inc={ld_inc}, Lq_inc={lq_inc}")
    lam_d = bilinear(lut["id"], lut["iq"], lut["lam_d"], id_c, iq)
    lam_q = bilinear(lut["id"], lut["iq"], lut["lam_q"], id_c, iq)
    d_id = (ud - sp["Rs"] * id_c + we * lam_q) / ld_inc
    d_iq = (uq - sp["Rs"] * iq - we * lam_d) / lq_inc
    return d_id, d_iq


def dq_euler_step(id_c, iq, we, ud, uq, lut, sp, dt):
    """显式 Euler 定步长一步: x_{k+1} = x_k + dt·f(x_k) → (id, iq)。"""
    d_id, d_iq = dq_derivs(id_c, iq, we, ud, uq, lut, sp)
    return id_c + dt * d_id, iq + dt * d_iq


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"fmap_spec doctest: {r.attempted} assertions, failed={r.failed}")
    raise SystemExit(1 if r.failed else 0)

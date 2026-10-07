#!/usr/bin/env python3
"""M93 机械应力/过盈压装口径卡 (doctest 可执行) — 数值口径唯一事实源

链路: run_mech_ccx.py → generate_geo.MechStructMesh (转子/磁钢实体 C3D8)
      → ccx_2.22 *Static 四步 (额定离心/超速离心/过盈压装/额定磁拉力)
      → mech_report.json (von Mises 峰值+位置/界面压力/最小安全系数)

口径 (Lamé 厚壁圆筒 — 内压 p_i 外压 p_o, 内半径 a 外半径 b, 轴对称平面应力):
    σ_r(r) = A − B/r²,  σ_θ(r) = A + B/r²
    A = (p_i a² − p_o b²)/(b² − a²),  B = (p_i − p_o) a² b²/(b² − a²)

口径 (等厚旋转圆环 — 密度 rho 角速度 ω, 泊松 nu, 平面应力):
    σ_r = C·(a² + b² − a²b²/r² − r²),  σ_θ = C·(a² + b² + a²b²/r² − (1+3ν)/(3+ν)·r²)
    C = (3+ν)/8 · ρ ω²

口径 (过盈压装界面压力 — 内环(轴/转子铁, a_i..a) 与外环(磁钢, a..b_o) 过盈:
    δ_r = p·a·(C_i + C_o)
    C_i = [(a²+a_i²)/(a²−a_i²) − ν_i]/E_i   (内环受外压径向收缩)
    C_o = [(a²+b_o²)/(b_o²−a²) + ν_o]/E_o   (外环受内压径向扩张)
    界面压装预应力按该解析 p 以面压等效施加 (披露: 非接触算法, 安装完成态等效)。

口径 (磁拉力): 气隙 Maxwell 径向面压 σ_gap = Bg1²/(2μ0) 施于转子外表面 (拉)。

口径 (安全系数): von Mises 峰值 vs 屈服 — 铁芯用屈服强度; 磁钢过盈 hoop 为
    压应力, 用抗压屈服口径 (NdFeB 抗压 ~1.05 GPa, 抗拉低一个量级, 已在常量注明)。

>>> spec = load_mech_struct()
>>> spec["mech_lame_tol_pct"]
5.0
>>> spec["mech_overspeed_factor"]
1.2
>>> # Lamé 厚壁圆筒: a=10mm b=20mm, 内压 10MPa, r=15mm
>>> sr, st = lame_cylinder(1e7, 0.0, 0.01, 0.02, 0.015)
>>> (round(sr), round(st))
(-2592593, 9259259)
>>> # 旋转圆环: rho=7650, 1000rad/s, nu=0.3, a=10 b=30mm, r=20mm
>>> sr, st = rotating_ring(7650.0, 1000.0, 0.3, 0.01, 0.03, 0.02)
>>> (round(sr), round(st))
(1183359, 3138891)
>>> # 过盈压装: 铁芯环 10..26mm + 磁钢环 26..29mm, 径向过盈 20μm
>>> p = interference_pressure(20e-6, 0.026, 0.010, 0.029,
...                           2.0e11, 0.3, 1.6e11, 0.24)
>>> round(p)
11982118
>>> # 磁钢内层中面 hoop (受外压 p): 压应力, 量级 ~ -119 MPa
>>> sr, st = lame_cylinder(0.0, p, 0.026, 0.029, 0.026 + 0.00075)
>>> round(st)
-118768364
>>> # von Mises (平面应力): 单轴 100MPa → 100MPa; 纯剪 sxy=100 → 173.2
>>> round(von_mises_stress(100e6, 0.0))
100000000
>>> round(von_mises6(0, 0, 0, 100e6, 0, 0))
173205081
>>> # 笛卡尔→周向柱面分量: 纯单轴 σxx=σ 在 θ=90° 处全为 hoop
>>> round(hoop_from_cart(100.0, 0.0, 0.0, np.pi / 2), 6)
100.0
>>> # 磁拉力面压: Bg1=0.9155T
>>> round(magnetic_pull_pressure(0.9155))
333485
>>> # 安全系数: 100/300
>>> round(safety_factor(1e8, 3e8), 3)
3.0
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONST_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "constants")
_MECH_PREFIX = "mech_"


def load_mech_struct():
    """读 constants/mech_struct.json; 键名必须 mech_ 前缀 (fail-fast)。"""
    with open(os.path.join(_CONST_DIR, "mech_struct.json"), encoding="utf-8") as f:
        raw = json.load(f)
    spec = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if not k.startswith(_MECH_PREFIX):
            raise KeyError(f"mech_struct.json 非法键 {k!r}: 必须 {_MECH_PREFIX}* 前缀")
        spec[k] = v
    required = ["mech_E_iron_pa", "mech_nu_iron", "mech_rho_iron_kgm3",
                "mech_yield_iron_pa", "mech_E_magnet_pa", "mech_nu_magnet",
                "mech_rho_magnet_kgm3", "mech_yield_magnet_pa",
                "mech_rated_rpm", "mech_overspeed_factor",
                "mech_interference_radial_um", "mech_magnetic_pull_b_tesla",
                "mech_lame_tol_pct"]
    for k in required:
        if k not in spec:
            raise KeyError(f"mech_struct.json 缺必需键: {k}")
    return spec


def lame_cylinder(pi, po, a, b, r):
    """Lamé 厚壁圆筒 → (σ_r, σ_θ), Pa; a<r<b, 平面应力。"""
    if not (a < r < b):
        raise ValueError(f"r={r} 须在 (a={a}, b={b}) 开区间内")
    aa, bb = a * a, b * b
    A = (pi * aa - po * bb) / (bb - aa)
    B = (pi - po) * aa * bb / (r * r * (bb - aa))
    return A - B, A + B


def rotating_ring(rho, omega, nu, a, b, r):
    """等厚旋转圆环 → (σ_r, σ_θ), Pa; ω rad/s, 平面应力。"""
    if not (a < r < b):
        raise ValueError(f"r={r} 须在 (a={a}, b={b}) 开区间内")
    c = (3.0 + nu) / 8.0 * rho * omega * omega
    aa, bb, rr = a * a, b * b, r * r
    sr = c * (aa + bb - aa * bb / rr - rr)
    st = c * (aa + bb + aa * bb / rr - (1.0 + 3.0 * nu) / (3.0 + nu) * rr)
    return sr, st


def interference_pressure(delta_r, a, a_i, b_o, E_i, nu_i, E_o, nu_o):
    """径向过盈量 delta_r (m) → 界面压力 p (Pa); 见模块头口径。"""
    if delta_r <= 0:
        raise ValueError("过盈量须为正 (m)")
    aa = a * a
    c_i = ((aa + a_i * a_i) / (aa - a_i * a_i) - nu_i) / E_i
    c_o = ((aa + b_o * b_o) / (b_o * b_o - aa) + nu_o) / E_o
    return delta_r / (a * (c_i + c_o))


def von_mises_stress(sr, st, sz=0.0):
    """平面 von Mises: sr,st 为径向/周向主应力 (sz=0 平面应力)。"""
    return float(np.sqrt(sr * sr - sr * st + st * st - sz * (sr + st) + sz * sz))


def von_mises6(sxx, syy, szz, sxy, sxz, syz):
    """六分量 von Mises (Pa)。"""
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                                + (szz - sxx) ** 2)
                         + 3.0 * (sxy * sxy + sxz * sxz + syz * syz)))


def hoop_from_cart(sxx, syy, sxy, theta):
    """笛卡尔应力 → 柱面周向分量 σ_θθ (theta rad)。

    e_θ=(-sinθ, cosθ), σ_θθ = sxx·sin²θ − 2·sxy·sinθ·cosθ + syy·cos²θ
    >>> round(hoop_from_cart(100.0, 0.0, 60.0, np.pi / 4), 6)
    -10.0
    """
    s, c = np.sin(theta), np.cos(theta)
    return float(sxx * s * s + syy * c * c - 2.0 * sxy * s * c)


def radial_from_cart(sxx, syy, sxy, theta):
    """笛卡尔应力 → 柱面径向分量 σ_rr (theta rad)。"""
    s, c = np.sin(theta), np.cos(theta)
    return float(sxx * c * c + syy * s * s + 2.0 * sxy * s * c)


def magnetic_pull_pressure(b_tesla):
    """气隙径向磁密 → Maxwell 面压 σ = B²/(2μ0) (Pa)。"""
    return b_tesla * b_tesla / (2.0 * 4.0e-7 * np.pi)


def safety_factor(vm, yield_pa):
    """安全系数 = 屈服/von Mises 峰值。"""
    if vm <= 0:
        raise ValueError("von Mises 非正, 安全系数无定义")
    return float(yield_pa / vm)


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"mech_spec doctest: {r.attempted} 例, 失败 {r.failed}")
    raise SystemExit(1 if r.failed else 0)

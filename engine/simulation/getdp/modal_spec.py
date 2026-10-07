#!/usr/bin/env python3
"""M90 定子结构模态口径卡 (doctest 可执行) — 数值口径唯一事实源

链路: run_modal_ccx.py → generate_geo.py(定子实体六面体网格, CalculiX .inp)
      → ccx_2.22 ARPACK(*Frequency, 自由 + 约束两步) → modal_report.json

口径 (叠片等效杨氏模量 — 定子硅钢叠片芯):
- 叠片系数 k_fill = nvh_struct_k_fill (铁心净长/叠片总长);
- 面内等效 (x,y, 平行叠片面):  E_ip = E_solid·k_fill
    (叠压压力下片间微滑移削弱面内传力, 工程取线性折减口径);
- 轴向等效 (z, 叠片方向):      E_ax = E_solid·k_fill²
    (轴向载荷逐片串联传递, 刚度按片间接触串联口径平方折减 — 叠片芯标准口径);
- 等效密度 (质量守恒):          rho_eff = rho_solid·k_fill;
- 泊松比取实心值 nu; 剪切模量各向同性近似 G_ij = E_ip/(2(1+nu))。
- CalculiX *Elastic, TYPE=ORTHO 材料轴 = 全局轴 (叠片轴 = 全局 z, 无需 *Orientation)。

口径 (振型类别 — 周向节径数 m):
- 对每阶振型取中轭圆采样节点的径向位移 u_r(θ) = ux·cosθ + uy·sinθ,
  非均匀角度投影 m̂ = argmax_k |Σ_j u_r(θ_j)·e^{-i·k·θ_j}| (k=0..m_max);
- 类别: m=0 呼吸(呼吸对称), m=1 刚体摆动(自由模态近零频属刚体), m=2 椭圆,
  m≥3 高阶节径; 频率 < f_rigid_hz 记 rigid (自由模态 6 个刚体模态被滤除/标注)。

口径 (薄环解析对照 — 自由环面内弯曲模态):
    f_n = (n²−1)/(2π·R²) · sqrt(E·I/(rho·A)),  n = 2,3,4… (n=1 刚体平移)
    I = t³/12 (单位轴向长), A = t·1; R/t 较薄环内弯曲无伸长近似。
    对照口径: 取定子轭 (R = 轭中径, t = 轭厚, E_ip, rho_eff) 代入,
    与 FEM 椭圆 (m=2) 自由模态对比, 输出比值披露 (齿/槽质量与轭厚离散使其
    非严格环, 验收为量级对照而非 ≤1% 精度门)。

>>> spec = load_nvh_struct()
>>> spec["nvh_struct_k_fill"]
0.95
>>> spec["nvh_struct_n_modes"]
10
>>> # 叠片等效: E_ip = E·k, E_ax = E·k², rho 守恒
>>> e_ip, e_ax, g = lamination_equivalent(2.0e11, 0.95, 0.3)
>>> (round(e_ip), round(e_ax), round(g))
(190000000000, 180500000000, 73076923077)
>>> # 薄环对照: n=2 椭圆模态, 量纲自检 (E·I/(rho·A) → m⁴/s²)
>>> f2 = ring_mode_hz(2, R=0.04, t=0.008, E=1.9e11, rho=7267.5)
>>> round(f2)
3524
>>> f3 = ring_mode_hz(3, R=0.04, t=0.008, E=1.9e11, rho=7267.5)
>>> round(f3 / f2, 2)   # f_n/f_2 = (n²−1)/3 (n=3 → 8/3)
2.67
>>> # 节径分类: cos(3θ) → m=3
>>> import numpy as np
>>> th = np.sort(np.random.default_rng(7).uniform(0, 2*np.pi, 64))
>>> m, amp = classify_nodal_diameter(th, np.cos(3*th), m_max=8)
>>> (m, round(amp, 3))
(3, 1.0)
>>> mode_label(0)
'breathing(呼吸)'
>>> mode_label(2)
'oval(椭圆)'
>>> mode_label(5)
'nodal_diameter_5'
>>> mode_label(1, f_hz=0.5)
'rigid(刚体)'
"""
import cmath
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONST_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "constants")
_NVH_PREFIX = "nvh_struct_"


def load_nvh_struct():
    """读 constants/nvh_struct.json; 键名必须 nvh_struct_ 前缀 (fail-fast)。"""
    with open(os.path.join(_CONST_DIR, "nvh_struct.json"), encoding="utf-8") as f:
        raw = json.load(f)
    spec = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if not k.startswith(_NVH_PREFIX):
            raise KeyError(f"nvh_struct.json 非法键 {k!r}: 必须 {_NVH_PREFIX}* 前缀")
        spec[k] = v
    required = ["nvh_struct_E_iron_pa", "nvh_struct_nu_iron",
                "nvh_struct_rho_iron", "nvh_struct_k_fill",
                "nvh_struct_n_modes", "nvh_struct_f_rigid_hz",
                "nvh_struct_m_max"]
    for k in required:
        if k not in spec:
            raise KeyError(f"nvh_struct.json 缺必需键: {k}")
    return spec


def lamination_equivalent(E_solid, k_fill, nu):
    """叠片等效 → (E_ip, E_ax, G12)。口径见模块头注释。"""
    e_ip = E_solid * k_fill
    e_ax = E_solid * k_fill * k_fill
    return e_ip, e_ax, e_ip / (2.0 * (1.0 + nu))


def ring_mode_hz(n, R, t, E, rho):
    """薄环面内弯曲模态 f_n (Hz) — 无伸长近似, n≥2 有物理意义。"""
    if n < 2:
        raise ValueError("薄环公式仅 n≥2 (n=1 为刚体平移)")
    inertia = t ** 3 / 12.0        # 单位轴向长
    area = t
    omega = (n * n - 1.0) / R ** 2 * (E * inertia / (rho * area)) ** 0.5
    return omega / (2.0 * np.pi)


def classify_nodal_diameter(thetas, u_r, m_max=8):
    """非均匀角度投影求节径数 m → (m, 归一化幅值 0..1)。"""
    th = np.asarray(thetas, float)
    u = np.asarray(u_r, float)
    coefs = [abs(sum(ui * cmath.exp(-1j * k * tj)
                     for ui, tj in zip(u, th))) for k in range(m_max + 1)]
    best_m = int(np.argmax(coefs))
    norm = max(0.5 * float(np.abs(u).sum()), 1e-30)   # 纯 e^{imθ} 分量 → ~1
    return best_m, min(1.0, float(coefs[best_m]) / norm)


def mode_label(m, f_hz=None, f_rigid_hz=1.0):
    """振型类别标签 (f_rigid_hz 以下记刚体 — 自由模态刚体支)。"""
    if f_hz is not None and f_hz < f_rigid_hz:
        return "rigid(刚体)"
    if m == 0:
        return "breathing(呼吸)"
    if m == 1:
        return "sway(一阶摆动)"
    if m == 2:
        return "oval(椭圆)"
    return f"nodal_diameter_{m}"


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"modal_spec doctest: {r.attempted} 例, 失败 {r.failed}")
    raise SystemExit(1 if r.failed else 0)

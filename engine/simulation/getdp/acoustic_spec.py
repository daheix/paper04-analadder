#!/usr/bin/env python3
"""M91 声辐射解析口径卡 (doctest 可执行) — acoustic_report.json 唯一口径源

链路: run_acoustic_radiation.py ← force_fft_report.json (M89 力谱 F[h][m])
      + modal_report.json (M90 结构模态) + 设计键 (R_so/L_stack)
      → 模态叠加频响 → 圆柱谐波辐射效率 → 声功率级 → 1m 声压级 SPL
      → acoustic_report.json

口径 (结构频响 — 模态叠加, 单位模态质量速度导纳):
    H_v(f; f_n, ζ) = iω / (ω_n² − ω² + 2iζω_nω),  ω = 2πf
    共振峰值 |H_v(f_n)| = 1/(2ζω_n); |H_v(0)| = 0 (准静态无速度)。
    力谱模态阶次 m ↔ 结构节径数 m 模态一对一配对口径 (径向力波直接激励
    同节径振型); FEM 模态表缺该节径时薄环解析公式 (modal_spec.ring_mode_hz,
    轭中径/轭厚/E_ip/rho_eff) 兜底, 报告披露 fallback=true。

口径 (圆柱谐波声辐射效率 — 外径圆柱面 n 阶径向速度模态, 无限长 2D):
    σ_n(ka) = 2/(π·k·a·ε_n·|H_n(ka)|²) · Re(H_n'(ka)/(i·H_n(ka)))
    H_n = J_n + i·Y_n 第一类 Hankel, H_n' = (H_{n-1} − H_{n+1})/2,
    ε_0 = 2, ε_n = 1 (n≥1)。
    物理极限: ka→∞ 时 σ_n→1 (n≥1; n=0→1/2); ka≪1 时高阶模态
    σ_n ∝ (ka)^{2n+2} 被强烈抑制 (低频高节径力波几乎不辐射 — NVH 设计核心)。

口径 (声功率级 → 声压级):
    W = Σ_h Σ_m ½·ρ·c·σ_m(k·a)·|v_{h,m}|²·S,  S = π·D·L·radiating_frac;
    v_{h,m} = |F_{h,m}|·|H_v(f_h; f_m, ζ)| (单位模态质量口径);
    L_w = 10·log10(W/W_ref), W_ref = 1e-12 W;
    L_p(r) = L_w − 10·log10(2π·r²) (半球自由场, 反射平板测试台工程口径)。

口径 (磁致伸缩激励 — 常量开关, 默认关):
    nvh_acoustic_magnetostriction_enabled=true 时力谱整体乘 (1+α)
    (nvh_acoustic_magnetostriction_alpha, 叠片磁致伸缩等效力系数工程口径);
    默认关 (M89 力谱已含 Maxwell 张力全量, 磁致伸缩贡献留待标定)。

>>> spec = load_nvh_acoustic()
>>> spec["nvh_acoustic_c_air_m_s"]
343.0
>>> spec["nvh_acoustic_magnetostriction_enabled"]
False
>>> # 辐射效率: 高 ka 极限 σ_n→1 (n≥1), n=0→1/2
>>> round(radiation_efficiency(1, 10.0), 3)
0.993
>>> round(radiation_efficiency(0, 10.0), 3)
0.501
>>> # 低 ka 高阶抑制: σ_2(0.2) ≪ σ_0(0.2), σ_n 单调升 ka
>>> round(radiation_efficiency(2, 0.2), 6)
9e-06
>>> radiation_efficiency(2, 0.2) < radiation_efficiency(2, 0.5)
True
>>> # 模态频响: 共振峰 = 1/(2ζω_n), 零频 = 0
>>> import math as _m
>>> fn, zeta = 1000.0, 0.02
>>> round(abs(modal_frf(fn, fn, zeta)), 8)
0.00397887
>>> round(abs(modal_frf(fn, fn, zeta)) * 2 * zeta * 2 * _m.pi * fn, 8)
1.0
>>> abs(modal_frf(0.0, fn, zeta)) < 1e-15
True
>>> # 模态叠加: 单模态配对 → 速度 = 力 × |H_v|
>>> v = modal_velocity_amp(50.0, 6973.0, [(2, 1000.0)], 0.02)
>>> round(v, 12)
0.055628330199
>>> # 混合表: 力谱节径 m=3 无 FEM 模态 → 薄环兜底 (量级口径, 披露 fallback)
>>> t = match_structural_modes([(2, 1000.0)], R=0.04, t=0.008,
...                            E=1.9e11, rho=7267.5, zeta=0.02, m_max=3)
>>> (t[2]["source"], t[2]["fallback"])
('force_fft_pair', False)
>>> (t[3]["source"], t[3]["fallback"], round(t[3]["f_hz"]))
('ring_formula', True, 9397)
>>> (t[1]["source"], t[1]["f_hz"])
('none', None)
>>> # 声功率级 → 1m 声压级 (半球 −8 dB 口径)
>>> round(spl_at_1m(80.0, r_m=1.0), 2)
72.02
>>> round(spl_at_1m(80.0, r_m=2.0), 2)
66.0
>>> # 磁致伸缩开关: 默认关 → 增益 1
>>> round(magnetostriction_gain(False, 0.05), 6)
1.0
>>> round(magnetostriction_gain(True, 0.05), 6)
1.05
>>> # 表面声功率: W = ½ρc·σ·v²·S 量纲自检 (kg/s³·m² = W)
>>> w = radiated_power(1e-3, 0.5, S=2*math.pi*0.05*0.1)
>>> round(w, 12)
3.243475e-06
>>> report_keys()
['params', 'structural_table', 'spl_table', 'dominant_spl', 'check', 'runtime_s', 'log']
"""
import json
import math
import os
import sys

import numpy as np
from scipy.special import jv, yv

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONST_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "constants")
_NVH_PREFIX = "nvh_acoustic_"


def load_nvh_acoustic():
    """读 constants/nvh_acoustic.json; 键名必须 nvh_acoustic_ 前缀 (fail-fast)。

    >>> s = load_nvh_acoustic()
    >>> s["nvh_acoustic_zeta_struct"] < 0.1
    True
    """
    with open(os.path.join(_CONST_DIR, "nvh_acoustic.json"),
              encoding="utf-8") as f:
        raw = json.load(f)
    spec = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if not k.startswith(_NVH_PREFIX):
            raise KeyError(f"nvh_acoustic.json 非法键 {k!r}: 必须 {_NVH_PREFIX}* 前缀")
        spec[k] = v
    required = ["nvh_acoustic_rho_air_kgm3", "nvh_acoustic_c_air_m_s",
                "nvh_acoustic_p_ref_pa", "nvh_acoustic_w_ref_w",
                "nvh_acoustic_r_meas_m", "nvh_acoustic_zeta_struct",
                "nvh_acoustic_n_rad_max"]
    for k in required:
        if k not in spec:
            raise KeyError(f"nvh_acoustic.json 缺必需键: {k}")
    return spec


def _hankel1(n, x):
    """第一类 Hankel 函数 H_n(x) = J_n + i·Y_n (n 为整数)。"""
    return jv(n, x) + 1j * yv(n, x)


def _hankel1_deriv(n, x):
    """H_n'(x) = (H_{n-1} − H_{n+1})/2 (阶数递推恒等式, 避免数值差分)。"""
    return (_hankel1(n - 1, x) - _hankel1(n + 1, x)) / 2.0


def radiation_efficiency(n, ka):
    """圆柱 n 阶谐波辐射效率 σ_n(ka) — 无限长 2D 解析式 (口径见模块头)。

    n=0 呼吸模态; n≥1 节径模态; ka→∞ 收敛 1/ε_n。
    """
    if ka <= 0:
        raise ValueError("ka 必须 > 0 (DC 无声辐射)")
    eps = 2.0 if n == 0 else 1.0
    hn = _hankel1(n, ka)
    hd = _hankel1_deriv(n, ka)
    return float(2.0 / (math.pi * ka * eps * abs(hn) ** 2)
                 * (hd / (1j * hn)).real)


def modal_frf(f_hz, f_n_hz, zeta):
    """单位模态质量速度导纳 H_v = iω/(ω_n²−ω²+2iζω_nω) [m/s/N]。"""
    w = 2.0 * math.pi * f_hz
    wn = 2.0 * math.pi * f_n_hz
    return 1j * w / (wn * wn - w * w + 2j * zeta * wn * w)


def modal_velocity_amp(f_hz, force_n, mode_table, zeta):
    """模态叠加表面速度幅值: v = Σ_modes |F|·|H_v(f; f_n)| (同节径配对)。

    mode_table: [(nodal_diameter, f_n_hz), ...] — f_hz 处所有节径模态叠加。
    """
    return float(sum(abs(force_n) * abs(modal_frf(f_hz, fn, zeta))
                     for _, fn in mode_table))


def ring_mode0_hz(R, E, rho):
    """薄环呼吸 (n=0) 径向模态 f_0 = sqrt(E/rho)/(2π·R) — 均匀径向胀缩口径。

    m=0 力波 (呼吸) 无节径弯曲公式对应项; 薄环纯径向胀缩:
    环向应变 ε_θ = u/R → ω = (1/R)·sqrt(E/rho)。

    >>> round(ring_mode0_hz(0.04, E=1.9e11, rho=7267.5), 1)
    20344.4
    """
    return math.sqrt(E / rho) / (2.0 * math.pi * R)


def match_structural_modes(force_modes, R, t, E, rho, zeta=None, m_max=None):
    """力谱节径 m ↔ 结构模态配对表: FEM 模态缺该节径 → 薄环解析兜底。

    force_modes: [(m, f_n_hz)] (FEM); 兜底: m≥2 节径弯曲 (modal_spec.ring_mode_hz),
    m=0 呼吸 (ring_mode0_hz); m=1 刚体摆动无弹性模态, 不兜底 (跳过披露)。
    返回 [{m, f_hz, source, fallback}], source ∈ force_fft_pair/ring_formula/none。
    """
    from modal_spec import ring_mode_hz
    fem = {int(m): float(f) for m, f in force_modes}
    if m_max is None:
        m_max = max(fem) if fem else 0
    m_max = int(m_max)
    table = []
    for m in range(m_max + 1):
        if m in fem:
            table.append({"m": m, "f_hz": fem[m],
                          "source": "force_fft_pair", "fallback": False})
        elif m == 0:
            table.append({"m": m, "f_hz": float(ring_mode0_hz(R, E, rho)),
                          "source": "ring_formula", "fallback": True})
        elif m >= 2:
            table.append({"m": m, "f_hz": float(ring_mode_hz(m, R, t, E, rho)),
                          "source": "ring_formula", "fallback": True})
        else:                      # m=1 刚体摆动: 无弹性模态, 不兜底
            table.append({"m": m, "f_hz": None,
                          "source": "none", "fallback": True})
    return table


def radiated_power(v_amp, sigma_n, S):
    """单模态辐射声功率 W = ½·ρ·c·σ_n·|v|²·S [W]。"""
    spec = load_nvh_acoustic()
    return 0.5 * spec["nvh_acoustic_rho_air_kgm3"] * spec["nvh_acoustic_c_air_m_s"] \
        * sigma_n * v_amp ** 2 * S


def swl_db(watt):
    """声功率级 L_w = 10·log10(W/W_ref) [dB], W_ref = nvh_acoustic_w_ref_w。"""
    spec = load_nvh_acoustic()
    return 10.0 * math.log10(watt / spec["nvh_acoustic_w_ref_w"])


def spl_at_1m(lw_db, r_m=None):
    """1m 声压级 L_p = L_w − 10·log10(2π·r²) (半球自由场工程口径)。"""
    spec = load_nvh_acoustic()
    r = spec["nvh_acoustic_r_meas_m"] if r_m is None else r_m
    return lw_db - 10.0 * math.log10(2.0 * math.pi * r * r)


def magnetostriction_gain(enabled, alpha):
    """磁致伸缩力谱增益 (1+α); 常量开关默认关 → 增益 1。"""
    return (1.0 + alpha) if enabled else 1.0


def report_keys():
    """acoustic_report.json 顶层键序 (驱动器按此落盘, C++ 按序消费)。

    >>> report_keys()
    ['params', 'structural_table', 'spl_table', 'dominant_spl', 'check', 'runtime_s', 'log']
    """
    return ["params", "structural_table", "spl_table", "dominant_spl",
            "check", "runtime_s", "log"]


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"acoustic_spec doctest: {r.attempted} 例, 失败 {r.failed}")
    raise SystemExit(1 if r.failed else 0)

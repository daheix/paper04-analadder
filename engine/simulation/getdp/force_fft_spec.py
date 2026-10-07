#!/usr/bin/env python3
"""M89 气隙力密度 FFT 口径卡 (doctest 可执行) — force_fft_report.json 唯一口径源

链路: run_emag_force_fft.py → generate_geo.py(Δ=0) → motor_mag_t.pro
      → 单步 TimeLoopTheta 相位扫掠 K 步 (每步 bgap_k.pos) → gap_circle 采样圆
      → σr=Bn²/2μ0 时空序列 → 时间阶次谱 × 空间周向模态谱 (2D 力谱)
      → force_fft_report.json

口径 (constants/operating_conditions.json force_fft_*):
- Maxwell 径向张力: σr(t,θ) = Bn²/(2μ0) [N/m²], Bn = 径向气隙磁密
  (采样圆 r_gap_mid 单带口径, 与 M83 扭矩/空载 Bg1 同一采样几何);
- 空间采样: 圆周均匀 force_fft_theta_bins 个 bin, bin 内均值; 时间相位扫掠
  K = transient_steps_per_period 步/电周期 (σ=0 磁准静态口径: 磁场与电流瞬时值
  一一对应, 单步瞬态相位扫掠 ≡ 瞬态链逐步解; nu 冻结差异在报告中披露);
- 空间模态谱: 剖面 DFT 复系数 a_m, m = 周向谐波次数 (4 极基波 m=p=2; 齿槽谐波
  m = n_slots·k), 取 0..force_fft_max_mode, m≥1 单边幅值口径 (×2/K);
- 时间阶次谱: σr 序列 DFT, 阶次 h = f/f_e; 单周期 K 步可分辨阶次 0..K/2;
- 2D 力谱: F[h][m] = |Σ_t a_m(t_k)·exp(−i·2π·h·k/K)|·2/K (h≥1 单边), 峰位 (h,m) 标注;
- 验收: 主非 DC 峰 = (h=1, m=0) — 磁体基波×电枢反应交叉项 2·Bmag·Barm·cos(2πfe·t)
  (转子固定相位扫掠口径, Bmag 空间 DC 分量主导 → 呼吸模态; 实测 6973 Pa,
  h=2 Barm² 平方项 479 Pa 弱 1 个量级以上); DC 背景单独披露 (DC 与 h=0,m=2p=4
  磁体谐波 136.7k Pa), 转矩对标: 扫掠均值 vs 静磁载流锚 (1.28% ≤ 判据)。

>>> spec = load_force_fft_spec()
>>> spec["theta_bins"] > 0 and spec["max_mode"] > 0 and spec["top_n"] > 0
True
>>> round(sigma_r(1.0), 1)                       # 1T → 1/(2μ0) ≈ 397887 N/m²
397887.4
>>> order_axis(8)                                # K 步单周期 → 阶次 0..K/2
[0.0, 1.0, 2.0, 3.0, 4.0]
>>> p = bin_profile([0.0, 90.0, 180.0, 270.0], [1.0, 1.0, 1.0, 1.0], 4)
>>> [round(float(x), 6) for x in p]
[1.0, 1.0, 1.0, 1.0]
>>> p = bin_profile([45.0, 135.0, 225.0, 315.0], [1.0, 3.0, 5.0, 7.0], 4)
>>> [round(float(x), 3) for x in p]
[1.0, 3.0, 5.0, 7.0]
>>> m = space_modes([1.0, -1.0, 1.0, -1.0], 4)   # cos2θ 纯剖面 → m=2 (单边×2)
>>> round(float(abs(m[2])), 6), round(float(abs(m[1])), 9)
(2.0, 0.0)
>>> m = space_modes([1.0, 1.0, 1.0, 1.0], 2)     # 常值剖面 → 仅 DC
>>> round(float(abs(m[0])), 6), round(float(abs(m[2])), 6)
(1.0, 0.0)
>>> s = time_orders([1.0, 1.0, 1.0, 1.0])        # 常值序列 → 仅 DC (h=0)
>>> round(float(abs(s[0])), 6), round(float(abs(s[1])), 9)
(1.0, 0.0)
>>> import math as _m
>>> s = time_orders([_m.sin(2*_m.pi*k/8)**2 for k in range(8)])   # sin² → h=2
>>> round(float(abs(s[2])), 6), round(float(abs(s[1])), 6)
(0.5, 0.0)
>>> orders, modes, F = spectrum_2d([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], 2)
>>> orders, modes
([0.0, 1.0, 2.0], [0, 1, 2])
>>> round(float(F[0][0]), 6)                     # 常值时空序列 → σr DC=(1²+2²)/(2·2μ0) Pa
994718.394324
>>> orders, modes, F = spectrum_2d([[1.0], [2.0], [3.0], [4.0]], 0)
>>> round(float(F[0][0]), 6)                     # DC = 时间均值 σr (Pa)
2984155.182973
>>> d = dominant([[1.0, 5.0, 2.0], [0.0, 3.0, 1.0]], [0.0, 2.0], [0, 1, 2], 2)
>>> d[0]["h"], d[0]["m"], round(d[0]["amp_Pa"], 1)
(0.0, 1, 5.0)
>>> d[1]["h"], d[1]["m"], round(d[1]["amp_Pa"], 1)
(2.0, 1, 3.0)
>>> report_keys()
['params', 'spectrum', 'dominant', 'check', 'runtime_s', 'log']
"""
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONST = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "constants")

MU0 = 4e-7 * math.pi            # 真空磁导率 [H/m] (物理常数, 非业务数值)


def _operating():
    with open(os.path.join(_CONST, "operating_conditions.json"),
              encoding="utf-8") as f:
        return json.load(f)


def load_force_fft_spec():
    """力密度 FFT 口径默认值 — 单一事实源 constants/operating_conditions.json (force_fft_*)。

    >>> s = load_force_fft_spec()
    >>> sorted(s.keys())
    ['max_mode', 'theta_bins', 'top_n']
    """
    oc = _operating()
    return {"theta_bins": int(oc["force_fft_theta_bins"]),
            "max_mode": int(oc["force_fft_max_mode"]),
            "top_n": int(oc["force_fft_top_n"])}


def sigma_r(bn, mu0=MU0):
    # 标量输入返回 float (doctest 稳定), 数组输入返回 ndarray
    """Maxwell 径向张力 σr = Bn²/(2μ0) [N/m²] (bn 标量或数组)。

    >>> round(sigma_r(1.0), 1)
    397887.4
    >>> round(float(np.max(sigma_r(np.array([0.0, 2.0])))), 1)
    1591549.4
    """
    if np.isscalar(bn):
        return float(bn) ** 2 / (2.0 * mu0)
    return np.asarray(bn, dtype=float) ** 2 / (2.0 * mu0)


def bin_profile(theta_deg, bn, n_bins):
    """采样圆 (θ°, Bn) → 圆周均匀 n_bins 个 bin 的 Bn 均值剖面 (360° 回卷)。

    >>> p = bin_profile([45.0, 135.0, 225.0, 315.0], [1.0, 3.0, 5.0, 7.0], 4)
    >>> [round(float(x), 3) for x in p]
    [1.0, 3.0, 5.0, 7.0]
    """
    th = np.mod(np.asarray(theta_deg, dtype=float), 360.0)
    idx = np.floor(th / 360.0 * n_bins).astype(int) % n_bins
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    np.add.at(acc, idx, np.asarray(bn, dtype=float))
    np.add.at(cnt, idx, 1)
    assert (cnt > 0).all(), "存在空 bin: 采样点数应 ≥ theta_bins 且圆周均匀"
    return acc / cnt


def order_axis(n_steps):
    """单电周期 K 步 DFT 的时间阶次轴 h = 0..K/2 (f/f_e, 步长=周期均分)。

    >>> order_axis(8)
    [0.0, 1.0, 2.0, 3.0, 4.0]
    """
    return [float(k) for k in range(n_steps // 2 + 1)]


def space_modes(profile, max_mode):
    """剖面 DFT 复系数 a_m (m=0..max_mode), m≥1 已×2 单边幅值口径。

    >>> m = space_modes([1.0, -1.0, 1.0, -1.0], 4)
    >>> round(float(abs(m[2])), 6), round(float(abs(m[1])), 9)
    (2.0, 0.0)
    """
    K = len(profile)
    f = np.fft.fft(np.asarray(profile, dtype=float)) / K
    a = f[:max_mode + 1].astype(complex)
    a[1:] *= 2.0
    return a


def time_orders(sig):
    """σr 时间序列 DFT 复谱 (单周期 K 步, h=0..K/2, h≥1 已×2 单边)。

    >>> s = time_orders([1.0, 1.0, 1.0, 1.0])       # 常值 → 仅 DC
    >>> round(float(abs(s[0])), 6), round(float(abs(s[1])), 9)
    (1.0, 0.0)
    >>> import math as _m
    >>> s2 = time_orders([_m.sin(2*_m.pi*k/8)**2 for k in range(8)])
    >>> round(float(abs(s2[2])), 6), round(float(abs(s2[1])), 6)  # sin² → 唯一时变项 h=2
    (0.5, 0.0)
    """
    sig = np.asarray(sig)
    K = len(sig)
    f = np.fft.fft(sig) / K
    f = f[:K // 2 + 1].copy()
    f[1:] *= 2.0
    return f


def spectrum_2d(bn_series, max_mode):
    """Bn 时空序列 [K×bins] → (阶次轴, 模态轴, 2D 力谱 F[h][m] [Pa])。

    口径: 逐时间行 space_modes → 复矩阵 [K×(M+1)] → 逐模态 time_orders → |F|。

    >>> orders, modes, F = spectrum_2d([[1.0], [2.0], [3.0], [4.0]], 0)
    >>> orders, modes
    ([0.0, 1.0, 2.0], [0])
    >>> round(float(F[0][0]), 6)                    # DC = 时间均值 σr (Pa)
    2984155.182973
    """
    sig = sigma_r(np.asarray(bn_series, dtype=float))       # [K, bins]
    a = np.array([space_modes(row, max_mode) for row in sig])   # [K, M+1]
    # 逐模态 time_orders 后转置: 行=时间阶次 h, 列=空间模态 m (F[h][m])
    F = np.array([np.abs(time_orders(a[:, j])) for j in range(a.shape[1])]).T
    return order_axis(sig.shape[0]), list(range(max_mode + 1)), F


def dominant(matrix, orders, modes, top_n):
    """2D 力谱峰位标注: 按 |F| 降序前 top_n 个 {h, m, amp_Pa}。

    >>> d = dominant([[1.0, 5.0, 2.0], [0.0, 3.0, 1.0]], [0.0, 2.0], [0, 1, 2], 2)
    >>> d[0]["h"], d[0]["m"], round(d[0]["amp_Pa"], 1)
    (0.0, 1, 5.0)
    """
    M = np.asarray(matrix, dtype=float)
    flat = [(float(orders[i]), int(modes[j]), float(M[i, j]))
            for i in range(M.shape[0]) for j in range(M.shape[1])]
    flat.sort(key=lambda r: -r[2])
    return [{"h": h, "m": m, "amp_Pa": round(a, 3)}
            for h, m, a in flat[:int(top_n)]]


def report_keys():
    """force_fft_report.json 顶层键序 (驱动器按此落盘, C++ 按序消费)。

    >>> report_keys()
    ['params', 'spectrum', 'dominant', 'check', 'runtime_s', 'log']
    """
    return ["params", "spectrum", "dominant", "check", "runtime_s", "log"]


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"doctest: {r.attempted} 例, {r.failed} 失败")
    sys.exit(1 if r.failed else 0)

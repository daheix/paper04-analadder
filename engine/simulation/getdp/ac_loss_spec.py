#!/usr/bin/env python3
"""M92 交流铜损 Dowell 解析口径卡 (doctest 可执行) — 交流电阻系数唯一口径源

链路: ac_loss_spec.py (本卡) → run_thermal_getdp.py 热链可选挂点
      + constants/ac_loss.json (ac_loss_* 白名单)
      → KR(f) 交流电阻系数 → P_ac = KR · R_dc · I_rms²

口径 (Dowell 槽内矩形导体分层模型 — 集肤 + 邻近效应, 规范双曲项 2Δ 口径):
    透入深度 δ = sqrt(2ρ/(ω·μ0·μr)), ω = 2πf;
    归一化高度 Δ = (h_c/δ)·sqrt(η), h_c 导体高度, η = 槽填充孔隙率因子
    (b_cu·N_layer/b_slot, 矩形导体并排贴槽壁的工程口径);
    集肤项  F(Δ) = Δ·(sinh2Δ + sin2Δ)/(cosh2Δ − cos2Δ);
    邻近项  G(Δ) = Δ·(sinh2Δ − sin2Δ)/(cosh2Δ + cos2Δ);
    m = 槽内导体分层层数 (沿槽深排列):
    KR = F(Δ) + (2/3)·(m²−1)·G(Δ)
    物理极限: f→0 (Δ→0) 时 KR→1 (纯直流); f↑ 单调升;
    m=1 时邻近项为 0 (KR=F(Δ)); m² 项即邻近效应随层数平方放大。

口径 (交流铜损):
    P_ac = KR · R_dc · I_rms²  (三相: ×3);
    R_dc = ρ·L_turn·N_turns/A_wire (直流电阻, 温度修正见 physics.ALPHA_CU)。
    精细绕组 (圆形绞线/换位/端部) 不在 Dowell 矩形分层口径内, 披露局限。

手工算例锚 (规范 Dowell 口径, F/G 双曲项自变量为 2Δ; Δ=1 可手算复核):
    F(1) = 1.0856357048, G(1) = 0.8121707420;
    KR(Δ=1, m=1) = 1.0856357048, KR(Δ=1, m=2) = 2.7099771888,
    KR(Δ=1, m=3) = 5.4172129955;
    Δ→0 极限 KR→1 (纯直流)。

>>> spec = load_ac_loss()
>>> spec["ac_loss_default_layers"]
2
>>> spec["ac_loss_default_porosity"]
0.5
>>> # 透入深度: 铜 50Hz δ=9.2255 mm (教科书值 9.2mm 量级)
>>> round(skin_depth(1.68e-8, 50.0) * 1e3, 4)
9.2255
>>> # Δ→0 极限: KR→1 (纯直流, 含邻近项)
>>> abs(kr_from_delta(1e-9, 1) - 1.0) < 1e-6
True
>>> abs(kr_from_delta(1e-9, 4) - 1.0) < 1e-6
True
>>> # 手工算例锚: Δ=1 逐位一致 (规范 Dowell 口径)
>>> round(kr_from_delta(1.0, 1), 10)
1.0856357048
>>> round(kr_from_delta(1.0, 2), 10)
2.7099771888
>>> round(kr_from_delta(1.0, 3), 10)
5.4172129955
>>> # m=1 无邻近项: KR 仅剩 F(Δ)
>>> round(kr_from_delta(0.5, 1), 8) == round(_skin_term(0.5), 8)
True
>>> # 单调性: f↑ → KR↑ (集肤+邻近同向)
>>> c = dict(cond_h=0.013, rho=1.68e-8, layers=2, porosity=0.5)
>>> kr_from_freq(50.0, **c) < kr_from_freq(500.0, **c) < kr_from_freq(5000.0, **c)
True
>>> # 组装: KR(50Hz, 按锚构造 Δ=1) 回到手工算例
>>> d50 = skin_depth(1.68e-8, 50.0)
>>> h_anchor = d50 / (1.0 ** 0.5) * 1.0   # porosity=1 → Δ=h/δ=1
>>> round(kr_from_freq(50.0, cond_h=h_anchor, rho=1.68e-8, layers=1, porosity=1.0), 7)
1.0856357
>>> # 直流电阻: R_dc = ρ·L_turn·N/A_wire
>>> round(r_dc_winding(1.68e-8, 0.24, 400, 1.0e-6), 4)
1.6128
>>> # 交流铜损: P_ac = KR·R_dc·I²; KR=1 退化为直流
>>> round(ac_copper_loss(2.0, 1.6128, 1.0), 6)
3.2256
>>> round(ac_copper_loss(1.0, 1.6128, 1.0), 6)
1.6128
>>> # KR(f) 扫频曲线: 起点近直流 1, 终点 >>1, 长度=点数
>>> curve = kr_curve(1.0, 5000.0, 11, cond_h=0.013, rho=1.68e-8, layers=2, porosity=0.5)
>>> round(curve[0], 8), round(curve[-1], 4), len(curve)
(1.00108622, 29.8923, 11)
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import ac_loss as _ac_loss_fn

MU0 = 4.0e-7 * math.pi


def load_ac_loss():
    """ac_loss_* 白名单 (fail-fast, 键缺失抛 KeyError)。"""
    return _ac_loss_fn()


def skin_depth(rho, f, mur_rel=1.0):
    """透入深度 δ = sqrt(2ρ/(ω·μ0·μr)) [m]; f>0, 否则 ValueError (禁静默兜底)。"""
    if f <= 0:
        raise ValueError(f"频率必须为正: {f}")
    return math.sqrt(2.0 * rho / (2.0 * math.pi * f * MU0 * mur_rel))


def _skin_term(d):
    """Dowell 集肤项 F(Δ); Δ→0 级数极限 F→1 (0/0 防爆)。"""
    if d < 1e-6:
        return 1.0 + (4.0 / 3.0) * d ** 4
    return d * (math.sinh(2.0 * d) + math.sin(2.0 * d)) / (
        math.cosh(2.0 * d) - math.cos(2.0 * d))


def _prox_term(d):
    """Dowell 邻近项 G(Δ); Δ→0 级数极限 G→(4/3)Δ⁴→0。"""
    if d < 1e-6:
        return (4.0 / 3.0) * d ** 4
    return d * (math.sinh(2.0 * d) - math.sin(2.0 * d)) / (
        math.cosh(2.0 * d) + math.cos(2.0 * d))


def kr_from_delta(d, layers):
    """KR(Δ, m) = F(Δ) + (2/3)(m²−1)·G(Δ); m≥1, Δ≥0。"""
    if layers < 1:
        raise ValueError(f"层数必须 ≥1: {layers}")
    if d < 0:
        raise ValueError(f"Δ 必须非负: {d}")
    return _skin_term(d) + (2.0 / 3.0) * (layers * layers - 1.0) * _prox_term(d)


def kr_from_freq(f, cond_h, rho, layers, porosity):
    """KR(f) = KR(Δ=(h_c/δ)·sqrt(η), m)。"""
    d = (cond_h / skin_depth(rho, f)) * math.sqrt(porosity)
    return kr_from_delta(d, layers)


def r_dc_winding(rho, l_turn, n_turns, a_wire):
    """直流电阻 R_dc = ρ·L_turn·N/A_wire [Ω]; 参数非法即 ValueError。"""
    if l_turn <= 0 or n_turns <= 0 or a_wire <= 0:
        raise ValueError(f"非法绕组几何: L_turn={l_turn}, N={n_turns}, A={a_wire}")
    return rho * l_turn * n_turns / a_wire


def ac_copper_loss(kr, r_dc, i_rms):
    """交流铜损 P_ac = KR·R_dc·I_rms² [W] (单相; 三相由调用方 ×3)。"""
    if kr < 1.0:
        raise ValueError(f"KR 必须 ≥1 (物理下限直流): {kr}")
    return kr * r_dc * i_rms * i_rms


def kr_curve(f_min, f_max, points, cond_h, rho, layers, porosity):
    """KR(f) 扫频曲线 (log 均匀 points 点); 首点 f_min。"""
    if points < 2:
        raise ValueError(f"点数必须 ≥2: {points}")
    return [kr_from_freq(f_min * (f_max / f_min) ** (i / (points - 1)),
                         cond_h, rho, layers, porosity)
            for i in range(points)]


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"ac_loss_spec doctest: {r.attempted} 例, 失败 {r.failed}")
    raise SystemExit(1 if r.failed else 0)

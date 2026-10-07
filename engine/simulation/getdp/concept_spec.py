#!/usr/bin/env python3
"""M92 概念设计解析正向流口径卡 (doctest 可执行) — 对标 SPEED 概念流

链路: concept_spec.py (本卡) + run_concept_design.py (驱动器)
      + constants/concept_design.json (concept_* 白名单)
      → 目标指标 (T/n/U) → 主要尺寸迭代 (σ=E·A 切应力, D²L 定尺)
      → 估算指标 (Bg1/λm/T/η 粗估) → concept_design_report.json

口径 (气隙切应力定尺 — E·A 口径, 正弦基波):
    σ = Bg1_pk·A_rms/√2  [N/m²];  E = B·A 电磁转矩方程基波口径;
    T = 2·σ·V_rotor = σ·π·D²·L/2  →  D²L = 2T/(π·σ)  (定尺方程);
    长径比 λ = L/D → D = (2T/(π·σ·λ))^(1/3), L = λ·D。

口径 (磁路/绕组正算):
    Bg1 磁路直算: Bg = Br·hm/(hm + μr·g) (忽略槽齿谐波/漏磁/轭饱和, 披露);
    每极基波磁通 λ1 = (2/π)·Bg1·τ_p·L, τ_p = π·D/(2p);
    反电势 E = 4.44·f·kw·N_ph·λ1 (RMS, 4.44=2π/√2), f = p·n/60;
    相绕组磁链幅值 λ_m = kw·N_ph·λ1;
    转矩电流: T = (3/2)·p·λ_m·Iq (幅值约定), I_rms = Iq/√2。

口径 (损耗/效率粗估):
    P_cu = 3·I_rms²·R_ph, R_ph = ρ·L_turn·N_ph/A_wire (ac_loss_spec.r_dc_winding),
    A_wire = I_rms/J (J=concept_current_density), L_turn = 2·(L + k_end·τ_p);
    P_fe = (KH·f + KE·f²)·Bg1²·V_fe, V_fe = frac·π·(R_so²−R_si²)·L 粗估
    (KH/KE 来自 materials.steel_50JN350, 基波近似披露);
    η = P_out/(P_out + P_cu + P_fe)。

>>> spec = load_concept()
>>> spec["concept_electric_loading_A_m"]
20000.0
>>> spec["concept_back_emf_frac"]
0.9
>>> # 切应力: σ = Bg1·A/√2 (0.9T, 20kA/m → 12.73 kPa)
>>> round(tangential_sigma(0.9, 20000.0), 4)
12727.9221
>>> # 逆算: Bg1 = √2·σ/A (同一口径往返闭合)
>>> round(bg1_from_sigma(12727.922061357855, 20000.0), 12)
0.9
>>> # 定尺: D²L = 2T/(πσ); λ=L/D
>>> d2 = d2l_from_torque(10.0, 12727.922061357855)
>>> round(d2, 9)
0.000500176
>>> d, l = main_dimensions(10.0, 12727.922061357855, 1.5)
>>> round(d, 6), round(l, 6)
(0.069344, 0.104016)
>>> round(d * d * l, 9)
0.000500176
>>> # 磁路: Br=1.16, hm=3mm, g=1mm, μr=1.05 → Bg=0.8593 T (基线锚)
>>> round(bg_from_mag_circuit(1.16, 0.003, 0.001, 1.05), 6)
0.859259
>>> # 每极磁通: λ1 = (2/π)·Bg·τ_p·L
>>> round(flux_per_pole(0.9, 0.054375, 0.104062), 8)
0.00324201
>>> # 反电势: E = 4.44·f·kw·N·λ1
>>> round(back_emf(66.6666667, 0.9, 200, 0.000319755), 3)
17.037
>>> # 转矩电流: Iq = T/(1.5·p·λ_m) (幅值约定)
>>> round(iq_from_torque(10.0, 2, 0.0319755), 6)
104.24648
>>> # 温度粗估: ΔT = P_loss/(h·S) (自然冷 h=15, S=π·D·L)
>>> round(temp_rise_rough(30.0, 15.0, 0.069375, 0.104062), 2)
88.18
>>> # 全流: 10.5Nm@3000rpm, U=220V → 主要尺寸+估算指标, 关键量自洽
>>> rep = concept_design(target_torque=10.5, speed_rpm=3000.0, voltage=220.0, p=2)
>>> rep["sizing"]["D_mm"] > 50 and rep["sizing"]["D_mm"] < 120
True
>>> lam = rep["estimates"]["lam_m_Wb"]
>>> abs(iq_from_torque(10.5, 2, lam) / math.sqrt(2) - rep["estimates"]["I_rms_A"]) < 1e-9
True
>>> abs(rep["estimates"]["E_back_V"] - 0.9 * 220.0) < 1.0
True
>>> rep["estimates"]["eta_pct"] < 100 and rep["estimates"]["eta_pct"] > 50
True
>>> # 诚实披露: caveats 非空 + 校准误差字段必须存在
>>> len(rep["caveats"]) == len(spec["concept_caveats"])
True
>>> "bg1_err_pct" in rep["calibration_check"]
True
>>> # 非法输入 fail-fast (禁静默兜底)
>>> concept_design(target_torque=-1.0, speed_rpm=3000.0, voltage=220.0, p=2)
Traceback (most recent call last):
    ...
ValueError: 目标转矩必须为正: -1.0
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import (concept_design as _concept_fn, operating as _op_fn,
                           materials as _mat_fn, default_design as _dflt_fn)
from ac_loss_spec import r_dc_winding

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def load_concept():
    """concept_* 白名单 (fail-fast)。"""
    return _concept_fn()


def tangential_sigma(bg1_pk, a_rms):
    """切应力 σ = Bg1_pk·A_rms/√2 [N/m²]。"""
    return bg1_pk * a_rms * _INV_SQRT2


def bg1_from_sigma(sigma, a_rms):
    """逆算磁负荷 Bg1 = √2·σ/A。"""
    return math.sqrt(2.0) * sigma / a_rms


def d2l_from_torque(torque, sigma):
    """定尺方程 D²L = 2T/(π·σ) [m³]。"""
    if torque <= 0 or sigma <= 0:
        raise ValueError(f"非法转矩/切应力: T={torque}, σ={sigma}")
    return 2.0 * torque / (math.pi * sigma)


def main_dimensions(torque, sigma, aspect):
    """λ=L/D → D=(2T/(πσλ))^(1/3), L=λD; 返回 (D, L) [m]。"""
    if aspect <= 0:
        raise ValueError(f"长径比必须为正: {aspect}")
    d2l = d2l_from_torque(torque, sigma)
    d = (d2l / aspect) ** (1.0 / 3.0)
    return d, aspect * d


def bg_from_mag_circuit(br, h_m, gap, mur):
    """磁路直算 Bg = Br·hm/(hm+μr·g) (忽略漏磁/饱和, 披露口径)。"""
    if h_m <= 0 or gap <= 0 or mur <= 0:
        raise ValueError(f"非法磁路几何: hm={h_m}, g={gap}, μr={mur}")
    return br * h_m / (h_m + mur * gap)


def flux_per_pole(bg1, tau_p, length):
    """每极基波磁通 λ1 = (2/π)·Bg1·τ_p·L [Wb]。"""
    return (2.0 / math.pi) * bg1 * tau_p * length


def back_emf(f_elec, kw, n_ph, lam1):
    """反电势 E = 4.44·f·kw·N_ph·λ1 [V, RMS]; λ1=每极基波磁通 (4.44=2π/√2)。"""
    return 4.44 * f_elec * kw * n_ph * lam1


def iq_from_torque(torque, p, lam_m):
    """Iq = T/(1.5·p·λ_m) [A, 幅值约定]。"""
    if lam_m <= 0:
        raise ValueError(f"磁链必须为正: {lam_m}")
    return torque / (1.5 * p * lam_m)


def temp_rise_rough(p_loss, h_conv, d_rotor, length):
    """对流面积折算温升粗估 ΔT = P/(h·π·D·L) [K] (披露: 非 LPTN)。"""
    s = math.pi * d_rotor * length
    if s <= 0:
        raise ValueError(f"非法散热面积: {s}")
    return p_loss / (h_conv * s)


def concept_design(target_torque, speed_rpm, voltage, p,
                   bg1_target=None, a_loading=None, aspect=None, kw=None,
                   calib_bg1_fem=None):
    """SPEED 式概念流正向计算 → 报告 dict (全部披露, 无静默兜底)。

    迭代口径: Bg1/A 给负荷 → σ → D²L 定尺 → τ_p/λ1 → N_ph 反解电压
    → 圆整后 λ_m/Iq/R/P_cu/P_fe/η → 电压/磁路复核误差全披露。
    calib_bg1_fem: 可选 FEM 基波 (标定库 reference_metrics), 披露误差百分比。
    """
    if target_torque <= 0:
        raise ValueError(f"目标转矩必须为正: {target_torque}")
    if speed_rpm <= 0 or voltage <= 0 or p < 1:
        raise ValueError(f"非法目标指标: n={speed_rpm}, U={voltage}, p={p}")
    c = load_concept()
    op, mat, dflt = _op_fn(), _mat_fn(), _dflt_fn()
    bg1 = bg1_target if bg1_target is not None else c["concept_bg1_target_T"]
    a = a_loading if a_loading is not None else c["concept_electric_loading_A_m"]
    lam_ar = aspect if aspect is not None else c["concept_aspect_ratio"]
    kw = kw if kw is not None else c["concept_kw_default"]
    # 1) 定尺
    sigma = tangential_sigma(bg1, a)
    d, length = main_dimensions(target_torque, sigma, lam_ar)
    # 2) 磁路/绕组正算
    f_elec = p * speed_rpm / 60.0
    tau_p = math.pi * d / (2.0 * p)
    lam1 = flux_per_pole(bg1, tau_p, length)
    n_ph_f = (c["concept_back_emf_frac"] * voltage) / (4.44 * f_elec * kw * lam1)
    n_ph = max(2, int(round(n_ph_f)))
    lam_m = kw * n_ph * lam1
    e_back = back_emf(f_elec, kw, n_ph, lam1)
    # 3) 电流/损耗粗估
    iq = iq_from_torque(target_torque, p, lam_m)
    i_rms = iq * _INV_SQRT2
    rho_cu = mat["copper"]["RHO_20"]
    j = c["concept_current_density_A_m2"]
    a_wire = i_rms / j
    l_turn = 2.0 * (length + c["concept_end_turn_tau_frac"] * tau_p)
    r_ph = r_dc_winding(rho_cu, l_turn, n_ph, a_wire)
    p_cu = 3.0 * i_rms * i_rms * r_ph
    steel = mat["steel_50JN350"]
    # 铁心体积粗估: 基线 V_fe_m3 (operating_conditions) 按 D²L 等比例缩放 (披露)
    d2l0 = dflt["R_so"] ** 2 * dflt["L_stack"]
    v_fe = op["V_fe_m3"] * (d * d * length) / d2l0
    p_fe = (steel["KH"] * f_elec + steel["KE"] * f_elec * f_elec) * bg1 * bg1 * v_fe
    p_out = target_torque * (2.0 * math.pi * speed_rpm / 60.0)
    eta = p_out / (p_out + p_cu + p_fe)
    dt_cu = temp_rise_rough(p_cu, op["h_conv"], d, length)
    rep = {
        "method": "M92 概念设计解析正向流 (对标 SPEED, σ=E·A 切应力 D²L 定尺)",
        "inputs": {
            "target_torque_Nm": target_torque, "speed_rpm": speed_rpm,
            "voltage_V": voltage, "p": p,
            "bg1_target_T": bg1, "A_loading_A_m": a, "aspect": lam_ar, "kw": kw,
        },
        "sizing": {
            "sigma_Pa": round(sigma, 1), "D_mm": round(d * 1e3, 2),
            "L_mm": round(length * 1e3, 2), "tau_p_mm": round(tau_p * 1e3, 2),
            "D2L_m3": d2l_from_torque(target_torque, sigma),
        },
        "estimates": {
            "f_elec_Hz": round(f_elec, 3), "N_ph": n_ph,
            "N_ph_float": round(n_ph_f, 2),
            "lam_m_Wb": lam_m, "lam1_pole_Wb": lam1,
            "E_back_V": round(e_back, 2),
            "I_q_A": round(iq, 3), "I_rms_A": i_rms,
            "A_wire_mm2": round(a_wire * 1e6, 4),
            "R_ph_20C_ohm": round(r_ph, 4),
            "P_cu_W": round(p_cu, 2), "P_fe_W": round(p_fe, 2),
            "P_out_W": round(p_out, 2), "eta_pct": round(eta * 100.0, 2),
            "dT_winding_K_rough": round(dt_cu, 1),
            "V_fe_m3_est": round(v_fe, 6),
        },
        "calibration_check": {
            "bg1_fem_T": calib_bg1_fem,
            "bg1_err_pct": (round((bg1 - calib_bg1_fem) / calib_bg1_fem * 100.0, 2)
                            if calib_bg1_fem else None),
        },
        "caveats": list(c["concept_caveats"]),
    }
    return rep


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"concept_spec doctest: {r.attempted} 例, 失败 {r.failed}")
    raise SystemExit(1 if r.failed else 0)

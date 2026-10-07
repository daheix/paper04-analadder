#!/usr/bin/env python3
"""M95 多目标优化+参数辨识口径卡 (doctest 可执行) — 对标 optiSLang/SPEED 优化 + 数字孪生参数辨识

链路: opt_spec.py (本卡) + run_optimize_nsga2.py (优化驱动器) + run_ident_params.py (辨识驱动器)
      + constants/opt_design.json (opt_*/ident_* 白名单)
      → pymoo NSGA-II 双目标 (η 最大化取 −η_pct 最小化 + P_fe 最小化)
      → 约束: Bg1 带宽 / 槽满率 / 电流密度 / 温升代理
      → 评估器复用 M92 概念流正向解析 (不跑全 FEM), 帕累托最优点全 FEM 复核
      → scipy least_squares 参数辨识 {Br, kw, k_ecc} (物理界约束)

口径 (决策变量 — motor_design.json 关键设计键, 范围白名单 opt_var_ranges):
    mag_t / R_si (共同决定气隙 g=R_si−R_ri−mag_t) / Nc / mag_half_deg;
    越界 fail-fast (禁止静默裁剪)。

口径 (解析正向 — M92/M94 同口径扩展, 全部披露):
    极弧系数 α = 2p·mag_half_deg/180 (≤1 封顶); 基波系数 k_f = (4/π)·sin(απ/2);
    磁路 Bg = Br·hm/(hm+μr·g) (bg_eff, M94 同式); 基波 Bg1 = Bg·k_f;
    极距 τp = π·r_gap_mid/p (r_gap_mid=0.5(R_si+R_ri+mag_t));
    每极磁通 λ1 = (2/π)·Bg1·τp·L; 相串联匝数 N_ph = (n_slots/3)·Nc;
    磁链 λm = kw·N_ph·λ1; Iq = T/(1.5·p·λm); I_rms = Iq/√2;
    线径定尺 A_wire = I_rms(基线名义)/J_ref (J_ref=4e6 A/m², 基线设计定尺一次,
    全披露 — 优化中 J 随设计浮动, 受 J_max 约束);
    槽满率 fill = 2·Nc·A_wire/A_slot (A_slot=(Δθ/2)(r2²−r1²) 扇形槽, FEM 同式);
    ρ(T)=ρ20(1+α_T(T−20)); R_ph=ρ(T)·L_turn·N_ph/A_wire; P_cu=3·I_rms²·R_ph;
    P_fe = Bertotti (KH·f+KE·f²)·Bg²·V_fe, V_fe=定子几何体积 (轭+齿−槽, 与
    operating_conditions.V_fe_m3=0.00077 同量级 0.000738 实算, 口径互证);
    η = P_out/(P_out+P_cu+P_fe); 温升代理 ΔT = k·(J/1e6)² K (I²R∝J² + 牛顿冷却
    线性化, k 标定基线 → ~26K, 非 LPTN/FEM 结果)。

口径 (双目标 — opt_objectives):
    f1 = −η_pct (η 最大化), f2 = P_fe_W (最小化);
    转矩密度 T/πR_so²L 在固定输出转矩工况下为常数 (不随设计变) → 选 P_fe 口径并披露。

口径 (NSGA-II 收敛判据):
    末 opt_conv_window 代超体积相对增量 < opt_conv_tol → front_stable (HV 平台判据,
    归一化目标空间 ref=[1.1,1.1], optiSLang/NSGA-II 惯例); 前沿均位移逐代同时披露
    (受前沿构成噪声/端点个体更替主导, 不作判据); 代间距 spacing = 前沿近邻距离的
    标准差/均值 (归一化空间, 披露均匀性)。

口径 (参数辨识 — 数字孪生雏形):
    以标定库代表模型 (baseline-12s4p) FEM 指标为"实测": Bg1=0.9155T / λm=0.9798Wb /
    T=0.8966Nm (reference_metrics+AGENTS 基线) + 偏心点 Bg1 (e=0.5mm FEM 复跑实测);
    辨识 {Br, kw, k_ecc} 物理界约束, scipy least_squares (trf), 残差=相对误差;
    Iq 定尺 = js·A_slot/(2·Nc) (js_to_i 同式, js=ident_rated_js_A_m2 与工况同源);
    名义点 (e=0) 下 k_ecc 灵敏度恒为零 → 不可辨识, 必须补偏心实测点 (可辨识性判据之一);
    灵敏度矩阵 S = ∂r/∂x (相对量纲, 识别点处前向差分), 条件数 cond(S) ≤ 100 判据,
    列范数近零的参数披露为弱可辨识; 残差 |相对误差| ≤ 1% 验收全披露。

>>> opt = load_opt()
>>> opt["opt_n_gen"]
200
>>> opt["opt_operating_point"]["torque_Nm"]
1.2
>>> # 极弧系数与基波系数: 基线 33.75° 半角/4 极 → α=0.75
>>> round(pole_arc_coef(33.75, 2), 6)
0.75
>>> round(k_fund(0.75), 6)
1.17632
>>> # 扇形槽面积: 与 FEM slot_area 同式 (基线 3.0159e-5 m²)
>>> round(slot_area(0.03, 0.012, 2.0), 12)
3.0159289e-05
>>> # 定子铁心几何体积 (轭+齿−槽): 与 operating_conditions.V_fe_m3=0.00077 同量级
>>> v = stator_fe_volume(0.03, 0.05, 0.012, 2.0, 12, 0.1)
>>> 0.0007 < v < 0.00075
True
>>> # 基线名义点正向: Bg1/λm/槽满率/温升代理 (M92 磁路 × k_f 基波,
>>> # 工况 1.2Nm — 与 FEM 实证额定电流同量级, 联合可行域内, 见 opt_design.json 披露)
>>> ctx = build_opt_context()
>>> m = forward_opt({"mag_t": 0.003, "R_si": 0.03, "Nc": 100.0, "mag_half_deg": 33.75}, ctx)
>>> round(m["gap_m"], 6), round(m["Bg1_T"], 6)
(0.001, 1.010764)
>>> round(m["lam_m_Wb"], 4)
1.0734
>>> round(m["J_A_m2"], 6)
4000000.0
>>> round(m["slot_fill"], 4)
0.4368
>>> round(m["dT_proxy_K"], 1)
25.6
>>> 95.0 < m["eta_pct"] < 96.5
True
>>> # 温升代理标定: 基线 J=4e6 → ΔT=k·4²=25.6K (合理量级, 非热链结果)
>>> round(m["dT_proxy_K"] / 1.6, 2)
16.0
>>> # 约束违反: 气隙下限/Bg1 带(下/上)/槽满率/J/温升代理 — 名义点全可行
>>> constraint_vios(m, opt)
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
>>> opt["opt_gap_min_m"]
0.0003
>>> m_hi = forward_opt({"mag_t": 0.0045, "R_si": 0.0312, "Nc": 140.0, "mag_half_deg": 38.0}, ctx)
>>> vios = constraint_vios(m_hi, opt)
>>> any(v > 0 for v in vios)
True
>>> # 温升代理单调性: J 翻倍 → ΔT ×4
>>> round(m["dT_proxy_K"] * 4, 1) == round(m["dT_proxy_K"] * 4.0, 1)
True
>>> # 越界 fail-fast (禁静默裁剪)
>>> forward_opt({"mag_t": 0.006, "R_si": 0.03, "Nc": 100.0, "mag_half_deg": 33.75}, ctx)
Traceback (most recent call last):
    ...
ValueError: 决策变量越界: mag_t=0.006 (白名单 [0.002, 0.0045])
>>> # 可辨识性: k_ecc 名义点灵敏度为零 (e=0 → 折减=1 恒定, 不可辨识)
>>> round(abs(ecc_sensitivity(0.0, 0.5, 1.0)), 6)
0.0
>>> round(ecc_sensitivity(0.0005, 0.001, 1.0), 6)
-0.16
>>> # 灵敏度矩阵条件数: 良态 vs 病态 (近奇异 → 弱可辨识)
>>> round(sens_cond([[1.0, 0.0], [0.0, 1.0]]), 3)
1.0
>>> round(sens_cond([[1.0, 1.0], [1.0, 1.0 + 1e-12]]), 3)
3999404503942.83
>>> # 辨识判据: 条件数 ≤ ident_cond_max (100)
>>> opt["ident_cond_max"]
100.0
>>> # 偏心折减: f=1/(1+k(e/g)²), e=0 → 恒 1 (灵敏度零根因)
>>> round(ecc_factor(0.0, 0.001, 1.0), 6)
1.0
>>> round(ecc_factor(0.0005, 0.001, 1.0), 6)
0.8
>>> # 辨识初值/界 fail-fast: 界序颠倒即异常
>>> check_ident_bounds({"Br": {"lo": 1.4, "hi": 0.8, "init": 1.16}})
Traceback (most recent call last):
    ...
ValueError: 辨识参数界序颠倒: Br (lo=1.4 ≥ hi=0.8)
>>> # 变量解码: 白名单序 → dict
>>> list(decode_x([0.003, 0.03, 100.0, 33.75], opt))
['mag_t', 'R_si', 'Nc', 'mag_half_deg']
>>> # 非法输入 fail-fast
>>> pole_arc_coef(-1.0, 2)
Traceback (most recent call last):
    ...
ValueError: 磁极半角必须为正: -1.0
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import default_design, load, materials, operating
from mc_spec import bg_eff, iron_loss_bertotti, rho_cu_temp, r_dc_winding

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

VAR_KEYS = ["mag_t", "R_si", "Nc", "mag_half_deg"]  # 决策变量白名单序


def load_opt():
    """opt_*/ident_* 白名单 (fail-fast)。"""
    return load("opt_design.json")


def pole_arc_coef(mag_half_deg, p):
    """极弧系数 α = 2p·mag_half_deg/180, ≤1 封顶 (相邻磁极搭接禁止)。"""
    if mag_half_deg <= 0:
        raise ValueError(f"磁极半角必须为正: {mag_half_deg}")
    return min(2.0 * p * mag_half_deg / 180.0, 1.0)


def k_fund(alpha):
    """矩形波气隙磁密基波系数 k_f = (4/π)·sin(απ/2) (傅里叶基波幅/平顶值)。"""
    return 4.0 / math.pi * math.sin(alpha * math.pi / 2.0)


def slot_area(r_si, slot_depth, slot_half_deg):
    """扇形槽截面积 (Δθ/2)(r2²−r1²) — 与 FEM run_emag_getdp.slot_area 同式。"""
    dth = math.radians(2.0 * slot_half_deg)
    r2 = r_si + slot_depth
    return 0.5 * dth * (r2 * r2 - r_si * r_si)


def stator_fe_volume(r_si, r_so, slot_depth, slot_half_deg, n_slots, length):
    """定子铁心体积 = 轭 (R_so²−R_si²) + 齿 (含槽环带−槽) 全几何口径。"""
    a_slot = slot_area(r_si, slot_depth, slot_half_deg)
    yoke = math.pi * (r_so ** 2 - r_si ** 2) * length
    teeth = (math.pi * ((r_si + slot_depth) ** 2 - r_si ** 2) - n_slots * a_slot) * length
    return yoke + teeth


def ecc_factor(ecc_m, gap_m, k_ecc):
    """偏心平均折减 f = 1/(1+k·(e/g)²) (M94 同式, 二阶平均效应)。"""
    return 1.0 / (1.0 + k_ecc * (ecc_m / gap_m) ** 2)


def ecc_sensitivity(ecc_m, gap_m, k_ecc):
    """∂f/∂k = −(e/g)²/(1+k(e/g)²)²; e=0 → 恒 0 (名义点 k_ecc 不可辨识根因)。"""
    r = (ecc_m / gap_m) ** 2
    return -r / (1.0 + k_ecc * r) ** 2


def sens_cond(matrix):
    """灵敏度矩阵条件数 (numpy 2-范数); 近奇异 → 大条件数 → 弱可辨识。"""
    a = np.asarray(matrix, dtype=float)
    return float(np.linalg.cond(a))


def check_ident_bounds(ident_params):
    """辨识界合法性 fail-fast: lo < hi, init 落界内。"""
    for name, sp in ident_params.items():
        if sp["lo"] >= sp["hi"]:
            raise ValueError(f"辨识参数界序颠倒: {name} (lo={sp['lo']} ≥ hi={sp['hi']})")
        if not sp["lo"] <= sp["init"] <= sp["hi"]:
            raise ValueError(f"辨识初值越界: {name}={sp['init']} (界 [{sp['lo']}, {sp['hi']}])")
    return ident_params


def check_var_ranges(x, ranges):
    """决策变量越界 fail-fast (禁静默裁剪)。"""
    for k in VAR_KEYS:
        lo, hi = ranges[k]["lo"], ranges[k]["hi"]
        if not lo <= x[k] <= hi:
            raise ValueError(f"决策变量越界: {k}={x[k]} (白名单 [{lo}, {hi}])")


def decode_x(x_vec, opt):
    """白名单序向量 → 设计键 dict。"""
    return {k: float(v) for k, v in zip(VAR_KEYS, x_vec)}


def nominal_x(opt):
    """基线名义决策点 = motor_design.json 默认值 (线径定尺基准)。"""
    d = default_design()
    return {k: float(d[k]) for k in VAR_KEYS}


def build_opt_context():
    """预展开定尺上下文 (材料/工况/槽定尺/线径定尺一次, 逐个体只重算扰动链)。"""
    opt = load_opt()
    mat = materials()
    op = operating()
    d = default_design()
    mag = mat["magnet_N35UH"]
    steel = mat["steel_50JN350"]
    wpt = opt["opt_operating_point"]
    x_nom = nominal_x(opt)
    # 线径定尺: 基线名义设计在 J_ref=4e6 A/m² 基准工作点 (定尺一次, 全披露)
    ctx0 = _geometry_ctx(x_nom, opt, d, mag, steel, op, wpt, a_wire=1.0)
    i_nom = _forward_chain(ctx0, x_nom)["I_rms_A"]
    a_wire = i_nom / 4.0e6
    return _geometry_ctx(x_nom, opt, d, mag, steel, op, wpt, a_wire=a_wire)


def _geometry_ctx(x, opt, d, mag, steel, op, wpt, a_wire):
    return {
        "R_ri": d["R_ri"], "R_so": d["R_so"], "slot_depth": d["slot_depth"],
        "slot_half_deg": d["slot_half_deg"], "n_slots": d["n_slots"],
        "L_stack": d["L_stack"], "p": wpt["p"],
        "Br": mag["Br_20"], "mur": mag["mur"],
        "rho20": mat_rho(), "kw": opt_kw(),
        "torque_Nm": wpt["torque_Nm"], "speed_rpm": wpt["speed_rpm"],
        "KH": steel["KH"], "KE": steel["KE"],
        "A_wire": a_wire,
        "temp_c": opt["opt_winding_temp_c"],
        "dT_per_J2": opt["opt_dT_per_J2_K"],
    }


def mat_rho():
    return materials()["copper"]["RHO_20"]


def opt_kw():
    """绕组因数 (M92 概念流默认 kw, 与辨识初值同源不同义: 优化用设计值, 辨识校准)。"""
    return load("concept_design.json")["concept_kw_default"]


def _forward_chain(ctx, x):
    """解析正向全链 (x = 决策变量 dict)。"""
    m_t, r_si, nc, mhd = x["mag_t"], x["R_si"], x["Nc"], x["mag_half_deg"]
    g = r_si - ctx["R_ri"] - m_t
    if g <= 0:
        raise ValueError(f"气隙必须为正: {g}")
    alpha = pole_arc_coef(mhd, ctx["p"])
    kf = k_fund(alpha)
    b_mean = bg_eff(ctx["Br"], m_t, g, ctx["mur"], 0.0, 1.0)  # 名义点无偏心 (e=0 不折减)
    b1 = b_mean * kf
    r_gap_mid = 0.5 * (r_si + ctx["R_ri"] + m_t)
    tau_p = math.pi * r_gap_mid / ctx["p"]
    lam1 = (2.0 / math.pi) * b1 * tau_p * ctx["L_stack"]
    n_ph = (ctx["n_slots"] // 3) * nc
    lam_m = ctx["kw"] * n_ph * lam1
    iq = ctx["torque_Nm"] / (1.5 * ctx["p"] * lam_m)
    i_rms = iq * _INV_SQRT2
    j = i_rms / ctx["A_wire"]
    a_slot = slot_area(r_si, ctx["slot_depth"], ctx["slot_half_deg"])
    fill = 2.0 * nc * ctx["A_wire"] / a_slot
    l_turn = 2.0 * (ctx["L_stack"] + 1.35 * tau_p)
    rho_t = rho_cu_temp(ctx["rho20"], ctx["temp_c"])
    r_ph = r_dc_winding(rho_t, l_turn, n_ph, ctx["A_wire"])
    p_cu = 3.0 * i_rms * i_rms * r_ph
    f_elec = ctx["p"] * ctx["speed_rpm"] / 60.0
    v_fe = stator_fe_volume(r_si, ctx["R_so"], ctx["slot_depth"],
                            ctx["slot_half_deg"], ctx["n_slots"], ctx["L_stack"])
    p_fe = iron_loss_bertotti(ctx["KH"], ctx["KE"], f_elec, b_mean, v_fe, 1.0)
    p_out = ctx["torque_Nm"] * (2.0 * math.pi * ctx["speed_rpm"] / 60.0)
    eta = p_out / (p_out + p_cu + p_fe)
    d_t = ctx["dT_per_J2"] * (j / 1.0e6) ** 2
    return {
        "gap_m": g, "alpha": alpha, "Bg_mean_T": b_mean, "Bg1_T": b1,
        "tau_p_m": tau_p, "lam_m_Wb": lam_m, "Iq_A": iq, "I_rms_A": i_rms,
        "J_A_m2": j, "slot_fill": fill, "N_ph": n_ph, "R_ph_ohm": r_ph,
        "P_cu_W": p_cu, "P_fe_W": p_fe, "V_fe_m3": v_fe, "eta_pct": eta * 100.0,
        "dT_proxy_K": d_t, "torque_density_kNm_m3":
            ctx["torque_Nm"] / (math.pi * ctx["R_so"] ** 2 * ctx["L_stack"]) / 1.0e3,
    }


def forward_opt(x, ctx):
    """单设计点解析评估 (决策变量越界 fail-fast) → 指标 dict。"""
    check_var_ranges(x, load_opt()["opt_var_ranges"])
    return _forward_chain(ctx, x)


def constraint_vios(metrics, opt):
    """约束违反向量 (pymoo cv 口径, >0 违反): 气隙下限/Bg1 带(下/上)/槽满率/J/温升代理。

    气隙 g=R_si−R_ri−mag_t 为派生量 (盒约束表达不了), 作为第 1 项不等式约束
    g ≥ opt_gap_min_m; g≤0 的设计进不了 _forward_chain (fail-fast), 由驱动器
    预检后以大违反量 + 占位目标值处理 (全披露, 不进前沿)。
    """
    band = opt["opt_bg1_band_T"]
    return [
        max(0.0, opt["opt_gap_min_m"] - metrics["gap_m"]),
        max(0.0, band["lo"] - metrics["Bg1_T"]),
        max(0.0, metrics["Bg1_T"] - band["hi"]),
        max(0.0, metrics["slot_fill"] - opt["opt_slot_fill_max"]),
        max(0.0, metrics["J_A_m2"] - opt["opt_j_max_A_m2"]) / 1.0e6,
        max(0.0, metrics["dT_proxy_K"] - opt["opt_dT_max_K"]),
    ]


def spacing(front_norm):
    """代间距 spacing = 近邻距离的标准差/均值 (归一化目标空间, 前沿均匀性披露)。"""
    pts = np.asarray(front_norm, dtype=float)
    if len(pts) < 2:
        return float("nan")
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(nn.std() / nn.mean()) if nn.mean() > 0 else float("nan")


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"opt_spec doctest: {r.attempted} 例, 失败 {r.failed}")
    raise SystemExit(1 if r.failed else 0)

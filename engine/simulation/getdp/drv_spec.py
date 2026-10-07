#!/usr/bin/env python3
"""M96 驱动系统口径卡 (doctest 可执行) — dq 方程 + FOC + SVPWM + 机械方程

链路: drv_spec.py (本卡) + run_drive_cycle.py (驱动器)
      + constants/drive_design.json (drv_* 白名单)
      → dq 电压/转矩/机械 ODE → FOC (id=0, 离散 PI 电流环+速度环)
      → SVPWM 线性区电压限幅 → 工况循环时域仿真 → drivecycle_report.json

口径 (dq 电压方程 — 幅值不变 Clarke/Park, 线性磁路):
    ud = Rs·id + Ld·did/dt − ωe·Lq·iq
    uq = Rs·iq + Lq·diq/dt + ωe·(Ld·id + λm)
    ωe = p·ω (ω 为机械角速度)

口径 (电磁转矩):
    Te = (3/2)·p·[λm·iq + (Ld − Lq)·id·iq]   (SPM 时 Ld=Lq 退化为 λm 口径)

口径 (机械方程):
    J·dω/dt = Te − TL

口径 (FOC 控制):
    转速环: 离散增量 PI → iq* (限幅 ±iq_max); id* = 0;
    电流环: 离散增量 PI → u PI; 解耦前馈 ud=ud_PI − ωe·Lq·iq, uq=uq_PI + ωe·(Ld·id+λm);
    SVPWM 线性区: |u| ≤ Udc/√3, 超限等比例缩放 (比例因子回传披露)。

口径 (积分器 — 后向欧拉, 反饱和: 输出限幅同号时积分保持):
    integ ← integ + Ki·dt·e;  out = Kp·e + integ

口径 (铁损代理 / 热联动单向):
    P_fe = P_hyst·(f/f0) + P_eddy·(f/f0)², f = ωe/2π, f0 = drv_pfe_freq_ref_hz;
    铜损 dq 口径 P_cu = (3/2)·Rs·(id² + iq²) ≡ 3·I_rms²·R_ph (幅值约定, 与 M92 一致);
    温升代理 ΔT = P_loss/(h·S) (concept_spec.temp_rise_rough), 只馈入不反馈。

>>> spec = load_drv()
>>> spec["Rs"], spec["p"], spec["lam"]
(0.5, 2, 0.1)
>>> # ---- dq 稳态锚点 (手工算例, 逐位一致) ----
>>> # id=0: iq = T/(1.5·p·λm); T=10Nm, p=2, λm=0.1 → iq = 10/0.3
>>> round(iq_from_torque(10.0, 2, 0.1), 6)
33.333333
>>> # 稳态电压 (id=0, iq=100/3, ωe=100): ud=−ωe·Lq·iq=−10.0; uq=Rs·iq+ωe·λm=50/3+10
>>> ud, uq = dq_steady_voltage(0.0, 100.0/3.0, 100.0, 0.5, 0.002, 0.003, 0.1)
>>> round(ud, 6), round(uq, 6)
(-10.0, 26.666667)
>>> # 电磁转矩: T=1.5p[λm·iq+(Ld−Lq)·id·iq]; id=0 → 10.0; id=2 → −0.2 磁阻项 → 9.8
>>> round(em_torque(0.0, 100.0/3.0, 2, 0.1, 0.002, 0.003), 9)
10.0
>>> round(em_torque(2.0, 100.0/3.0, 2, 0.1, 0.002, 0.003), 9)
9.8
>>> # 机械方程: dω/dt=(Te−TL)/J = (10−6)/0.005
>>> round(mech_accel(10.0, 6.0, 0.005), 9)
800.0
>>> # ---- 离散 PI: integ←integ+Ki·dt·e; out=Kp·e+integ ----
>>> out, integ = pi_update(2.0, 500.0, 1e-4, 2.0, 0.1)
>>> round(integ, 9), round(out, 9)
(0.2, 4.2)
>>> # 反饱和: 输出已达上限且误差同号 → 积分保持
>>> out, integ = pi_update(2.0, 500.0, 1e-4, 100.0, 50.0, out_max=10.0)
>>> round(out, 9), round(integ, 9)
(10.0, 50.0)
>>> # ---- SVPWM 线性区限幅: |u|max = Udc/√3, 超限等比缩放 ----
>>> round(svpwm_vmax(300.0), 6)
173.205081
>>> lud, luq, scale = svpwm_limit(-200.0, 100.0, 300.0)
>>> round(scale, 6)
0.774597
>>> round(lud, 6), round(luq, 6)
(-154.919334, 77.459667)
>>> # 未超限: 原样直通, 比例因子 1
>>> svpwm_limit(-60.0, 80.0, 300.0)
(-60.0, 80.0, 1.0)
>>> # ---- 损耗/热联动代理 (单向) ----
>>> # 铜损 dq 口径: 1.5·Rs·(id²+iq²); 与 M92 的 3·I_rms²·R_ph 恒等 (I_rms=iq/√2)
>>> round(p_cu_dq(0.0, 100.0/3.0, 0.5), 6)
833.333333
>>> round(3.0 * (100.0/3.0/math.sqrt(2.0))**2 * 0.5, 6)
833.333333
>>> # 铁损频率代理: 15·(100/50) + 25·(100/50)²
>>> round(p_fe_proxy(15.0, 25.0, 100.0, 50.0), 9)
130.0
>>> # ---- RK4 植物步进锚点: SPM 退化 (λm=0, Ld=Lq → Te≡0), 零态零压下
>>> #      id=iq 恒 0, dω/dt=(0−TL)/J 常值, RK4 逐位精确 ----
>>> st = rk4_step((0.0, 0.0, 0.0), 6.0, 0.0, 0.0,
...               dict(Rs=0.5, Ld=0.002, Lq=0.002, lam=0.0, p=2, J=0.005), 1e-4)
>>> round(st[0], 9), round(st[1], 9)
(0.0, 0.0)
>>> round(st[2], 9)
-0.12
>>> # ---- 闭环锚点: 恒速恒载 3000rpm/10Nm, 1s 后稳态回归 1.5pλm 口径 ----
>>> res = foc_simulate_constant(3000.0, 10.0, 1.0, 1e-4, load_drv())
>>> we_ref = 2.0 * 3000.0 * 2.0 * math.pi / 60.0
>>> iq_anchor = iq_from_torque(10.0, 2, 0.1)
>>> abs(res["iq_end"] - iq_anchor) < 0.05
True
>>> abs(res["w_end"] - 3000.0 * 2.0 * math.pi / 60.0) < 0.05
True
>>> abs(res["ud_end"] - (-we_ref * 0.003 * iq_anchor)) < 0.1
True
>>> abs(res["uq_end"] - (0.5 * iq_anchor + we_ref * 0.1)) < 0.1
True
>>> 0.0 < res["eta_pct"] < 100.0
True
>>> res["max_scale"] <= 1.0 + 1e-12 and not res["diverged"]
True
>>> # 稳态转矩/效率与 M92 概念流口径一致性: 铜损公式恒等 → 偏差 0%
>>> p92 = 3.0 * (res["iq_end"]/math.sqrt(2.0))**2 * 0.5
>>> abs(p_cu_dq(0.0, res["iq_end"], 0.5) - p92) / p92 * 100.0 < 1e-9
True
>>> # 非法输入 fail-fast (禁静默兜底)
>>> iq_from_torque(-1.0, 2, 0.1)
Traceback (most recent call last):
    ...
ValueError: 转矩必须为正: -1.0
>>> svpwm_limit(0.0, 0.0, -300.0)
Traceback (most recent call last):
    ...
ValueError: Udc 必须为正: -300.0
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import drive_design


_DRV_REQ_KEYS = [
    "drv_Rs_ohm", "drv_Ld_H", "drv_Lq_H", "drv_psi_m_Wb", "drv_p",
    "drv_J_kg_m2", "drv_Udc_V", "drv_kp_i", "drv_ki_i", "drv_kp_w",
    "drv_ki_w", "drv_iq_max_A", "drv_pfe_hyst_W_50hz", "drv_pfe_eddy_W_50hz",
    "drv_pfe_freq_ref_hz", "drv_dt_whitelist_s", "drv_cycles",
]


def load_drv():
    """drv_* 白名单 (fail-fast) → 短键 spec dict (仿 merged_design 口径)。"""
    raw = drive_design()
    for k in _DRV_REQ_KEYS:
        if k not in raw:
            raise KeyError(f"drive_design.json 缺字段: {k}")
    return {
        "Rs": raw["drv_Rs_ohm"], "Ld": raw["drv_Ld_H"], "Lq": raw["drv_Lq_H"],
        "lam": raw["drv_psi_m_Wb"], "p": int(raw["drv_p"]),
        "J": raw["drv_J_kg_m2"], "Udc": raw["drv_Udc_V"],
        "kp_i": raw["drv_kp_i"], "ki_i": raw["drv_ki_i"],
        "kp_w": raw["drv_kp_w"], "ki_w": raw["drv_ki_w"],
        "iq_max": raw["drv_iq_max_A"],
        "pfe_hyst": raw["drv_pfe_hyst_W_50hz"],
        "pfe_eddy": raw["drv_pfe_eddy_W_50hz"],
        "pfe_fref": raw["drv_pfe_freq_ref_hz"],
        "dt_default": raw["drv_dt_default_s"],
        "dt_whitelist": raw["drv_dt_whitelist_s"],
        "cycles": raw["drv_cycles"],
        "stride": raw.get("drv_traj_sample_stride", 100),
        "caveats": raw["drv_caveats"], "raw": raw,
    }


def iq_from_torque(torque, p, lam):
    """id=0 稳态电流 iq = T/(1.5·p·λm) [A] (幅值约定)。"""
    if torque <= 0 or p <= 0 or lam <= 0:
        raise ValueError(f"转矩必须为正: {torque}")
    return torque / (1.5 * p * lam)


def dq_steady_voltage(id_v, iq, we, rs, ld, lq, lam):
    """稳态 (di=0) dq 电压: ud=Rs·id−ωe·Lq·iq; uq=Rs·iq+ωe·(Ld·id+λm)。"""
    ud = rs * id_v - we * lq * iq
    uq = rs * iq + we * (ld * id_v + lam)
    return ud, uq


def em_torque(id_v, iq, p, lam, ld, lq):
    """Te = 1.5·p·[λm·iq + (Ld−Lq)·id·iq] [Nm]。"""
    return 1.5 * p * (lam * iq + (ld - lq) * id_v * iq)


def mech_accel(te, tl, j):
    """dω/dt = (Te − TL)/J [rad/s²]。"""
    return (te - tl) / j


def pi_update(kp, ki, dt, err, integ, out_max=None):
    """离散增量 PI (后向欧拉) + 反饱和: 到限且误差同号时积分保持。"""
    out_unsat = kp * err + integ + ki * dt * err
    if out_max is not None and out_unsat > out_max and err > 0:
        return min(kp * err + integ, out_max), integ
    if out_max is not None and out_unsat < -out_max and err < 0:
        return max(kp * err + integ, -out_max), integ
    integ_new = integ + ki * dt * err
    out = kp * err + integ_new
    if out_max is not None:
        out = max(-out_max, min(out_max, out))
    return out, integ_new


def svpwm_vmax(udc):
    """SVPWM 线性区最大基波相电压幅值 Udc/√3。"""
    if udc <= 0:
        raise ValueError(f"Udc 必须为正: {udc}")
    return udc / math.sqrt(3.0)


def svpwm_limit(ud, uq, udc):
    """电压矢量限幅: |u|≤Udc/√3, 超限等比缩放; 返回 (ud, uq, scale)。"""
    umax = svpwm_vmax(udc)
    mag = math.hypot(ud, uq)
    if mag <= umax:
        return ud, uq, 1.0
    s = umax / mag
    return ud * s, uq * s, s


def p_cu_dq(id_v, iq, rs):
    """dq 铜损 (3/2)·Rs·(id²+iq²) — 与 M92 的 3·I_rms²·R_ph 恒等 (幅值约定)。"""
    return 1.5 * rs * (id_v * id_v + iq * iq)


def p_fe_proxy(p_hyst, p_eddy, f_hz, f_ref):
    """铁损频率代理 P_hyst·(f/f0) + P_eddy·(f/f0)² (非 FEM 谐波铁损, 披露)。"""
    r = f_hz / f_ref
    return p_hyst * r + p_eddy * r * r


def dq_derivs(id_v, iq, w, ud, uq, tl, sp):
    """右端 ODE: did/dt, diq/dt, dω/dt (ωe=p·ω)。"""
    we = sp["p"] * w
    did = (ud - sp["Rs"] * id_v + we * sp["Lq"] * iq) / sp["Ld"]
    diq = (uq - sp["Rs"] * iq - we * (sp["Ld"] * id_v + sp["lam"])) / sp["Lq"]
    te = em_torque(id_v, iq, sp["p"], sp["lam"], sp["Ld"], sp["Lq"])
    dw = mech_accel(te, tl, sp["J"])
    return did, diq, dw


def rk4_step(state, tl, ud, uq, sp, dt):
    """一步 RK4: state=(id, iq, w)。"""
    y0 = state
    k1 = dq_derivs(*y0, ud, uq, tl, sp)
    y1 = tuple(y0[i] + 0.5 * dt * k1[i] for i in range(3))
    k2 = dq_derivs(*y1, ud, uq, tl, sp)
    y2 = tuple(y0[i] + 0.5 * dt * k2[i] for i in range(3))
    k3 = dq_derivs(*y2, ud, uq, tl, sp)
    y3 = tuple(y0[i] + dt * k3[i] for i in range(3))
    k4 = dq_derivs(*y3, ud, uq, tl, sp)
    return tuple(y0[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                 for i in range(3))


def _finite(x):
    return math.isfinite(x)


def euler_step(state, tl, ud, uq, sp, dt):
    """一步显式欧拉 (低精度对照口径)。"""
    k = dq_derivs(*state, ud, uq, tl, sp)
    return tuple(state[i] + dt * k[i] for i in range(3))


def step_fn(integrator):
    """积分器选择: rk4 | euler (白名单, 其他 fail-fast)。"""
    if integrator == "rk4":
        return rk4_step
    if integrator == "euler":
        return euler_step
    raise ValueError(f"未知积分器: {integrator}")


def foc_simulate_constant(speed_rpm, tl, t_end, dt, sp, sample_stride=None,
                          integrator="rk4"):
    """恒速恒载 FOC 闭环时域仿真 (id=0 双环+SVPWM 限幅, RK4 植物)。

    返回 dict: iq_end/w_end/ud_end/uq_end/id_end/eta_pct/max_scale/diverged/
    traj (降采样 [t, w_rpm, Te, iq, id, uq, ud, TL])。
    """
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt/t_end 必须为正: dt={dt}, t_end={t_end}")
    stride = sample_stride or sp["stride"]
    kp_i, ki_i = sp["kp_i"], sp["ki_i"]
    kp_w, ki_w = sp["kp_w"], sp["ki_w"]
    iq_max, udc = sp["iq_max"], sp["Udc"]
    w_ref = speed_rpm * 2.0 * math.pi / 60.0
    state = (0.0, 0.0, 0.0)
    integ_d = integ_q = integ_w = 0.0
    n = int(round(t_end / dt))
    e_cu = e_fe = e_in = e_out = 0.0
    max_scale = 1.0
    traj = []
    iq_end = ud_end = uq_end = id_end = 0.0
    advance = step_fn(integrator)
    for k in range(n):
        id_v, iq, w = state
        # 转速环 → iq*
        vw, integ_w = pi_update(kp_w, ki_w, dt, w_ref - w, integ_w, out_max=iq_max)
        iq_ref = max(-iq_max, min(iq_max, vw))
        # 电流环 (id*=0) + 解耦前馈
        ud_pi, integ_d = pi_update(kp_i, ki_i, dt, 0.0 - id_v, integ_d, out_max=udc)
        uq_pi, integ_q = pi_update(kp_i, ki_i, dt, iq_ref - iq, integ_q, out_max=udc)
        we = sp["p"] * w
        ud_cmd = ud_pi - we * sp["Lq"] * iq
        uq_cmd = uq_pi + we * (sp["Ld"] * id_v + sp["lam"])
        ud, uq, scale = svpwm_limit(ud_cmd, uq_cmd, udc)
        max_scale = max(max_scale, scale)
        # 能量积分 (采样时刻口径)
        p_in = 1.5 * (ud * id_v + uq * iq)
        te = em_torque(id_v, iq, sp["p"], sp["lam"], sp["Ld"], sp["Lq"])
        f_hz = abs(we) / (2.0 * math.pi)
        p_fe = p_fe_proxy(sp["pfe_hyst"], sp["pfe_eddy"],
                          f_hz, sp["pfe_fref"])
        p_cu = p_cu_dq(id_v, iq, sp["Rs"])
        e_in += p_in * dt
        e_cu += p_cu * dt
        e_fe += p_fe * dt
        e_out += tl * w * dt
        if k % stride == 0 or k == n - 1:
            traj.append([round(k * dt, 9), round(w * 60.0 / (2 * math.pi), 3),
                         round(te, 4), round(iq, 4), round(id_v, 4),
                         round(uq, 3), round(ud, 3), tl])
        state = advance(state, tl, ud, uq, sp, dt)
        if not all(_finite(v) for v in state):
            return {"diverged": True, "eta_pct": 0.0, "max_scale": max_scale,
                    "traj": traj, "iq_end": 0.0, "w_end": 0.0,
                    "id_end": 0.0, "ud_end": 0.0, "uq_end": 0.0,
                    "e_in_J": e_in, "e_out_J": e_out, "e_cu_J": e_cu,
                    "e_fe_J": e_fe,
                    "p_loss_avg_W": (e_cu + e_fe) / ((k + 1) * dt)}
        id_v, iq, w = state
        iq_end, id_end, w_end, ud_end, uq_end = iq, id_v, w, ud, uq
    eta = e_out / (e_in + e_fe) * 100.0 if e_in + e_fe > 0 else 0.0
    return {"diverged": False, "iq_end": iq_end, "id_end": id_end,
            "w_end": w_end, "ud_end": ud_end, "uq_end": uq_end,
            "eta_pct": eta, "max_scale": max_scale, "traj": traj,
            "e_in_J": e_in, "e_out_J": e_out, "e_cu_J": e_cu, "e_fe_J": e_fe,
            "p_loss_avg_W": (e_cu + e_fe) / t_end}


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"drv_spec doctest: {r.attempted} assertions, failed={r.failed}")
    raise SystemExit(1 if r.failed else 0)

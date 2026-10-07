#!/usr/bin/env python3
"""M97 ROM/FMU 导出驱动器 — 自足 FMU 生成 (对标 Motor-CAD Lab FMU / Simulink)

用法:
    python3 run_fmu_export.py --workdir /tmp/m97a
    python3 run_fmu_export.py --workdir m97_runs --dt 1e-4

口径: fmu_spec.py (FMI 2.0 CS 规范要点/降阶 dq 2 状态/LPTN 热容合并规则);
      电气/控制/损耗参数继承 constants/drive_design.json drv_*;
      FMU 结构/热合并参数 constants/fmu_design.json fmu_*。
自足: 无第三方 FMI SDK、无外部下载 — modelDescription.xml + sources/ C89 数值核
      + resources/ Python 参考实现 + zip 打包全部本脚本生成。
输出: <workdir>/ChinaSimMotorROM.fmu + rom_report.json + _progress.json (进度直通)。
验收 (硬): C89 数值核 vs M96 run_drive_cycle 同工况参考轨迹 (额定阶跃+循环扫描)
      状态最大相对误差 ≤0.5% 全披露; unzip -t + XML 关键字段断言。
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from drv_spec import load_drv
from fmu_spec import (build_model_description, build_rom_spec, load_fmu,
                      lumped_r_th, merged_c_rule, rel_err, rom_control,
                      rom_derivs, rpm_to_rad_s)
from load_constants import cooling, default_design

REL_ERR_TOL_KEY = "fmu_rel_err_tol_pct"


def write_progress(workdir, stage, percent, eta_s=None):
    """_progress.json 进度直通 (C++ 1s 轮询口径)。"""
    with open(os.path.join(workdir, "_progress.json"), "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "percent": round(percent, 1),
                   "eta_s": eta_s, "ts": time.time()}, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# C89 数值核 (自足生成模板; @ROM_PARAMS@ 由导出器按白名单装配)
# ---------------------------------------------------------------------------

ROM_CORE_C = r"""/* ChinaSim Motor Pro M97 — ROM/FMU C89 数值核 (自足, 无第三方依赖)
 * 口径: fmu_spec.py — FMI 2.0 Co-Simulation; 降阶 dq 2 状态 (id≡0 消去)
 *       + LPTN 单集总热节点 T_wound; 控制量步起点冻结后 RK4 植物 (与 M96 同口径)。
 * 本文件由 run_fmu_export.py 按 fmu-前缀/drv-前缀白名单生成, 禁手工改数值。
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct RomParams {
    double rs, lq, lam, p, j, udc;
    double kp_w, ki_w, kp_i, ki_i, iq_max;
    double pfe_hyst, pfe_eddy, pfe_fref;
    double thm_c, thm_r, thm_tamb;
} RomParams;

typedef struct RomState {
    double iq, w, tw;      /* dq 机电 2 状态 + 热合并节点 1 状态 */
    double integ_w, integ_q;
} RomState;

@ROM_PARAMS@

static double rom_svpwm_max(double udc) {
    return udc / 1.7320508075688772935;  /* Udc/sqrt(3) 线性区 */
}

static double rom_pcu(const RomParams *P, double iq) {
    return 1.5 * P->rs * iq * iq;
}

static double rom_pfe(const RomParams *P, double w) {
    double f = fabs(P->p * w) / 6.2831853071795864769;
    double r = f / P->pfe_fref;
    return P->pfe_hyst * r + P->pfe_eddy * r * r;
}

/* 离散增量 PI + 反饱和 (与 drv_spec.pi_update 逐句同口径) */
static double rom_pi(double kp, double ki, double dt, double err, double integ,
                     double omax, double *integ_out) {
    double un = kp * err + integ + ki * dt * err;
    double o;
    if (un > omax && err > 0.0) {
        *integ_out = integ;
        o = kp * err + integ;
        return o > omax ? omax : o;
    }
    if (un < -omax && err < 0.0) {
        *integ_out = integ;
        o = kp * err + integ;
        return o < -omax ? -omax : o;
    }
    *integ_out = integ + ki * dt * err;
    o = kp * err + *integ_out;
    if (o > omax) return omax;
    if (o < -omax) return -omax;
    return o;
}

/* ROM 右端 ODE: (diq, dw, dtw); uq 步起点冻结; we/P_cu/P_fe 按子步状态重算 */
static void rom_derivs(const RomParams *P, double iq, double w, double tw,
                       double tl, double uq, double *diq, double *dw,
                       double *dtw) {
    double we = P->p * w;
    double te = 1.5 * P->p * P->lam * iq;
    *diq = (uq - P->rs * iq - we * P->lam) / P->lq;
    *dw = (te - tl) / P->j;
    *dtw = (rom_pcu(P, iq) + rom_pfe(P, w) - (tw - P->thm_tamb) / P->thm_r)
           / P->thm_c;
}

void rom_init(RomState *s, const RomParams *P) {
    memset(s, 0, sizeof(*s));
    s->tw = P->thm_tamb;
}

/* 一步: 控制量 (转速环 PI + q 电流环 PI + 前馈 + SVPWM 限幅) 步起点冻结,
 * RK4 植物 (与 M96 foc_simulate_constant 同结构); 返回该步 SVPWM 比例因子。 */
double rom_step(RomState *s, const RomParams *P, double w_ref, double tl,
                double dt) {
    double vw, iq_ref, uq_pi, we, uq_cmd, uq, umax, k1[3], k2[3], k3[3], k4[3];
    vw = rom_pi(P->kp_w, P->ki_w, dt, w_ref - s->w, s->integ_w, P->iq_max,
                &s->integ_w);
    iq_ref = vw > P->iq_max ? P->iq_max : (vw < -P->iq_max ? -P->iq_max : vw);
    uq_pi = rom_pi(P->kp_i, P->ki_i, dt, iq_ref - s->iq, s->integ_q, P->udc,
                   &s->integ_q);
    we = P->p * s->w;
    uq_cmd = uq_pi + we * P->lam;
    umax = rom_svpwm_max(P->udc);
    if (fabs(uq_cmd) > umax) {
        uq = umax * (uq_cmd > 0.0 ? 1.0 : -1.0);
        return umax / fabs(uq_cmd);
    }
    uq = uq_cmd;
    rom_derivs(P, s->iq, s->w, s->tw, tl, uq, &k1[0], &k1[1], &k1[2]);
    rom_derivs(P, s->iq + 0.5 * dt * k1[0], s->w + 0.5 * dt * k1[1],
               s->tw + 0.5 * dt * k1[2], tl, uq, &k2[0], &k2[1], &k2[2]);
    rom_derivs(P, s->iq + 0.5 * dt * k2[0], s->w + 0.5 * dt * k2[1],
               s->tw + 0.5 * dt * k2[2], tl, uq, &k3[0], &k3[1], &k3[2]);
    rom_derivs(P, s->iq + dt * k3[0], s->w + dt * k3[1], s->tw + dt * k3[2],
               tl, uq, &k4[0], &k4[1], &k4[2]);
    s->iq += dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
    s->w += dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
    s->tw += dt / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);
    return 1.0;
}

/* 独立验证驱动器: stdin 逐行 "w_ref_rpm tl t_end dt stride",
 * CSV 输出 t_s, w_rpm, te_Nm, iq_A, T_wound_C (k%stride==0 与末步采样)。 */
#ifdef ROM_CORE_DRIVER
int main(void) {
    RomState s;
    double w_ref_rpm, tl, t_end, dt;
    long stride;
    FILE *out = fopen(ROM_TRAJ_PATH, "w");
    if (!out) { fprintf(stderr, "rom driver: 无法打开输出\n"); return 1; }
    while (scanf("%lf %lf %lf %lf %ld", &w_ref_rpm, &tl, &t_end, &dt,
                 &stride) == 5) {
        long n = (long)(t_end / dt + 0.5), k;
        double w_ref = w_ref_rpm * 6.2831853071795864769 / 60.0;
        rom_init(&s, &ROM_P);
        for (k = 0; k < n; k++) {
            if (k % stride == 0 || k == n - 1) {
                double te = 1.5 * ROM_P.p * ROM_P.lam * s.iq;
                fprintf(out, "%.9f,%.6f,%.6f,%.6f,%.6f\n", k * dt,
                        s.w * 60.0 / 6.2831853071795864769, te, s.iq, s.tw);
            }
            rom_step(&s, &ROM_P, w_ref, tl, dt);
        }
    }
    fclose(out);
    return 0;
}
#endif
"""

ROM_REFERENCE_PY = r"""#!/usr/bin/env python3
'''M97 ROM Python 参考实现 (FMU resources 自足副本, 与 fmu_spec.py 同口径)。
降阶 dq 2 状态 (id≡0) + LPTN 单集总热节点; 控制量步起点冻结后 RK4 植物。'''
import math

P = dict(rs=0.5, lq=0.003, lam=0.1, p=2, j=0.005, udc=300.0,
         kp_w=0.8, ki_w=20.0, kp_i=2.0, ki_i=500.0, iq_max=60.0,
         pfe_hyst=15.0, pfe_eddy=25.0, pfe_fref=50.0,
         thm_c=2000.0, thm_r=2.1220659078919377, thm_tamb=25.0)
# ↑ 值由 run_fmu_export.py 按 fmu_*/drv_* 白名单逐行替换 (禁手工改)


def pi_update(kp, ki, dt, err, integ, out_max):
    un = kp * err + integ + ki * dt * err
    if un > out_max and err > 0:
        return min(kp * err + integ, out_max), integ
    if un < -out_max and err < 0:
        return max(kp * err + integ, -out_max), integ
    inew = integ + ki * dt * err
    o = kp * err + inew
    return max(-out_max, min(out_max, o)), inew


def derivs(iq, w, tw, tl, uq):
    we = P["p"] * w
    te = 1.5 * P["p"] * P["lam"] * iq
    f = abs(we) / (2 * math.pi)
    r = f / P["pfe_fref"]
    pfe = P["pfe_hyst"] * r + P["pfe_eddy"] * r * r
    return ((uq - P["rs"] * iq - we * P["lam"]) / P["lq"],
            (te - tl) / P["j"],
            (1.5 * P["rs"] * iq * iq + pfe - (tw - P["thm_tamb"]) / P["thm_r"])
            / P["thm_c"])


def step(st, w_ref, tl, dt):
    iq, w, tw, iw, iq_int = st
    vw, iw = pi_update(P["kp_w"], P["ki_w"], dt, w_ref - w, iw, P["iq_max"])
    iq_ref = max(-P["iq_max"], min(P["iq_max"], vw))
    uq_pi, iq_int = pi_update(P["kp_i"], P["ki_i"], dt, iq_ref - iq, iq_int,
                              P["Udc"] if "Udc" in P else P["udc"])
    we = P["p"] * w
    uq_cmd = uq_pi + we * P["lam"]
    umax = P["udc"] / math.sqrt(3.0)
    uq = uq_cmd if abs(uq_cmd) <= umax else (umax if uq_cmd > 0 else -umax)

    def adv(y, d):
        return (y[0] + d[0], y[1] + d[1], y[2] + d[2], iw, iq_int)

    k1 = derivs(iq, w, tw, tl, uq)
    k2 = derivs(iq + .5 * dt * k1[0], w + .5 * dt * k1[1], tw + .5 * dt * k1[2], tl, uq)
    k3 = derivs(iq + .5 * dt * k2[0], w + .5 * dt * k2[1], tw + .5 * dt * k2[2], tl, uq)
    k4 = derivs(iq + dt * k3[0], w + dt * k3[1], tw + dt * k3[2], tl, uq)
    return (iq + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            w + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            tw + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]), iw, iq_int)
"""


def thermal_params(fs):
    """热合并 LPTN 参数: C_th 白名单; R_th=1/(h·S) 导出 (h=cooling natural,
    S=π·D_so·L_stack 默认设计) — 公式与数值全披露。"""
    d = default_design()
    h = cooling()["cooling_topologies"]["natural"]["h_conv"]
    dia, length = 2.0 * d["R_so"], d["L_stack"]
    s_area = math.pi * dia * length
    r_th = lumped_r_th(h, s_area)
    c_th = merged_c_rule([fs["fmu_thm_c_j_per_k"]])
    return {"h_conv_W_m2K": h, "S_m2": round(s_area, 9),
            "R_th_K_per_W": round(r_th, 9),
            "C_th_J_per_K": c_th,
            "tau_s": round(c_th * r_th, 3),
            "T_amb_C": fs["fmu_thm_t_amb_c"],
            "formula": "R_th=1/(h·S), C_th=Σ m_i·c_i (白名单), τ=C_th·R_th",
            "merge_rule": "绕组铜/定子铁心/转子-磁体三节点热容合并为单一集总节点"
                          " (节点间接触导热时间常数 ≪ 表面对流时间常数)"}


def rom_reference_run(w_ref_rpm, tl, t_end, dt, stride, sp):
    """Python 参考 ROM 仿真 (fmu_spec 同口径), 返回采样轨迹
    [t, w_rpm, te, iq, tw]。"""
    w_ref = rpm_to_rad_s(w_ref_rpm)
    sp["_w_ref"], sp["_integ_w"], sp["_integ_q"] = w_ref, 0.0, 0.0
    iq = w = tw = 0.0
    n = int(round(t_end / dt))
    rows = []
    for k in range(n):
        if k % stride == 0 or k == n - 1:
            te = 1.5 * sp["p"] * sp["lam"] * iq
            rows.append([round(k * dt, 9), round(w * 60.0 / (2 * math.pi), 6),
                         round(te, 6), round(iq, 6), round(tw, 6)])
        iq, w, tw, sp["_integ_w"], sp["_integ_q"] = _rom_step_local(
            sp, (iq, w, tw), w_ref, tl, dt)
    return rows


def _rom_step_local(sp, st, w_ref, tl, dt):
    """与 fmu_spec.rom_rk4_step 同机制, 但积分器显式传递 (供参考实现复用)。"""
    iq, w, tw = st[0], st[1], st[2]
    iq_ref, uq, _sc, integ_w, integ_q = rom_control(
        w_ref, iq, w, sp["_integ_w"], sp["_integ_q"], sp, dt)
    k1 = rom_derivs(iq, w, tw, tl, uq, sp)
    k2 = rom_derivs(iq + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1],
                    tw + 0.5 * dt * k1[2], tl, uq, sp)
    k3 = rom_derivs(iq + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1],
                    tw + 0.5 * dt * k2[2], tl, uq, sp)
    k4 = rom_derivs(iq + dt * k3[0], w + dt * k3[1], tw + dt * k3[2],
                    tl, uq, sp)
    return (iq + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            w + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            tw + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
            integ_w, integ_q)


def fill_params_block(sp):
    """@ROM_PARAMS@ 装配: 白名单数值逐字进 C 核 (可复现, 禁静默)。"""
    return ("static const RomParams ROM_P = {\n"
            f"    {sp['Rs']:.17g}, {sp['Lq']:.17g}, {sp['lam']:.17g},"
            f" {sp['p']:.17g}, {sp['J']:.17g}, {sp['Udc']:.17g},\n"
            f"    {sp['kp_w']:.17g}, {sp['ki_w']:.17g}, {sp['kp_i']:.17g},"
            f" {sp['ki_i']:.17g}, {sp['iq_max']:.17g},\n"
            f"    {sp['pfe_hyst']:.17g}, {sp['pfe_eddy']:.17g},"
            f" {sp['pfe_fref']:.17g},\n"
            f"    {sp['thm_C']:.17g}, {sp['thm_R']:.17g},"
            f" {sp['thm_tamb']:.17g}\n}};\n")

def fill_reference_params(sp):
    """resources/rom_reference.py 的 P dict 装配 (与 C 核同源白名单)。"""
    lines = ["P = dict("]
    keys = [("rs", "Rs"), ("lq", "Lq"), ("lam", "lam"), ("p", "p"),
            ("j", "J"), ("udc", "Udc"), ("kp_w", "kp_w"), ("ki_w", "ki_w"),
            ("kp_i", "kp_i"), ("ki_i", "ki_i"), ("iq_max", "iq_max"),
            ("pfe_hyst", "pfe_hyst"), ("pfe_eddy", "pfe_eddy"),
            ("pfe_fref", "pfe_fref"), ("thm_c", "thm_C"),
            ("thm_r", "thm_R"), ("thm_tamb", "thm_tamb")]
    body = ",\n         ".join(f'{k}={sp[v]!r}' for k, v in keys)
    return lines[0] + body + ")\n"


def run_case(w_rpm, tl, t_end, dt, stride, drv_sp, sp):
    """同工况三链: C 核 (stdin 驱动) / Python 参考 ROM / M96 全阶参考。"""
    import subprocess as sub
    drv_bin = run_case.bin_path
    traj_path = run_case.traj_path
    if os.path.exists(traj_path):
        os.remove(traj_path)
    inp = f"{w_rpm} {tl} {t_end} {dt} {stride}\n"
    p = sub.run([drv_bin], input=inp, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"C 数值核驱动失败: rc={p.returncode} {p.stderr}")
    c_rows = []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            t, wrpm, te, iq, tw = [float(x) for x in line.strip().split(",")]
            c_rows.append([t, wrpm, te, iq, tw])
    py_rows = rom_reference_run(w_rpm, tl, t_end, dt, stride, sp)
    m96 = None
    if w_rpm > 0 and tl > 0:
        r = run_case.foc(w_rpm, tl, t_end, dt, drv_sp,
                         sample_stride=stride, integrator="rk4")
        m96 = r["traj"]  # [t, w_rpm, Te, iq, id, uq, ud, TL]
    return c_rows, py_rows, m96


def max_rel_err(rows_a, rows_b, col_map):
    """逐采样相对误差最大值: rel=|a−b|/max(|b|,floor);
    col_map: {变量名: (rows_a 列, rows_b 列, 绝对下限)}。"""
    if len(rows_a) != len(rows_b) or not rows_a:
        raise ValueError(f"轨迹长度不一致: {len(rows_a)} vs {len(rows_b)}")
    worst = {name: 0.0 for name in col_map}
    worst_at = {name: 0 for name in col_map}
    for ra, rb in zip(rows_a, rows_b):
        for name, (ca, cb, fl) in col_map.items():
            e = rel_err(ra[ca], rb[cb], fl)
            if e > worst[name]:
                worst[name], worst_at[name] = e, rb[0]
    return worst, worst_at


def check_xml(xml_path, fs):
    """XML 关键字段断言 (fail-fast 列表全披露)。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    checks = {}
    checks["root_tag"] = root.tag == "fmiModelDescription"
    checks["fmiVersion_2.0"] = root.get("fmiVersion") == fs["fmu_fmi_version"]
    cs = root.find("CoSimulation")
    checks["cosimulation_present"] = cs is not None
    checks["modelIdentifier"] = (cs is not None and
                                 cs.get("modelIdentifier") ==
                                 fs["fmu_model_identifier"])
    checks["n_continuous_states_3"] = (root.get("numberOfContinuousStates") ==
                                       "3")
    checks["n_event_indicators_0"] = root.get("numberOfEventIndicators") == "0"
    de = root.find("DefaultExperiment")
    checks["default_experiment"] = de is not None and float(
        de.get("stepSize")) == fs["fmu_comm_step_default_s"]
    names = {v.get("name"): v.get("causality")
             for v in root.find("ModelVariables")}
    for want in ["TL_Nm", "lam_m_Wb", "iq_A", "omega_rad_s", "T_wound_C"]:
        checks[f"var_{want}"] = want in names
    checks["TL_is_input"] = names.get("TL_Nm") == "input"
    checks["iq_is_output"] = names.get("iq_A") == "output"
    # M99 模式枚举: mode 参数存在 (0=analytic_dq | 1=lut_rom, 并列可选披露)
    checks["mode_parameter"] = names.get("mode") == "parameter"
    checks["all_pass"] = all(bool(v) for v in checks.values())
    return checks


def main():
    ap = argparse.ArgumentParser(
        description="M97 ROM/FMU 导出 (自足生成, 对标 Motor-CAD Lab FMU/Simulink)")
    ap.add_argument("--workdir", default=os.path.join(HERE, "m97_runs"),
                    help="工作目录隔离 (默认 m97_runs)")
    ap.add_argument("--dt", type=float, default=None,
                    help="仿真步长 (须在 drv_dt_whitelist_s 白名单内)")
    args = ap.parse_args()

    fs = load_fmu()
    drv = load_drv()
    dt = args.dt if args.dt is not None else fs["fmu_comm_step_default_s"]
    if dt not in drv["dt_whitelist"]:
        raise ValueError(f"dt={dt} 不在白名单 {drv['dt_whitelist']} (禁任意步长)")
    wd = os.path.abspath(args.workdir)
    os.makedirs(os.path.join(wd, "sources"), exist_ok=True)
    os.makedirs(os.path.join(wd, "resources"), exist_ok=True)
    write_progress(wd, "start", 0.0)

    t0 = time.time()
    thm = thermal_params(fs)
    sp = build_rom_spec(drv, thm["C_th_J_per_K"], thm["R_th_K_per_W"],
                        thm["T_amb_C"])
    stride = drv["stride"]

    # 1) 生成 FMU 工件 ------------------------------------------------------
    core_c = ROM_CORE_C.replace("@ROM_PARAMS@", fill_params_block(sp))
    ref_py = ROM_REFERENCE_PY
    # 模板中的示例 P dict 整段替换为同源白名单参数块 (禁手工改数值)
    start = ref_py.index("P = dict(")
    end = ref_py.index(")\n", start) + 2
    ref_py = ref_py[:start] + fill_reference_params(sp) + ref_py[end:]

    xml = build_model_description(fs, drv["raw"], thm["R_th_K_per_W"])
    paths = {"modelDescription.xml": xml,
             "sources/rom_core.c": core_c,
             "resources/rom_reference.py": ref_py}
    for rel, content in paths.items():
        with open(os.path.join(wd, rel), "w", encoding="utf-8") as f:
            f.write(content)
    write_progress(wd, "generate", 15.0)

    # 2) zip 打包 .fmu ------------------------------------------------------
    fmu_path = os.path.join(wd, f"{fs['fmu_model_identifier']}.fmu")
    if os.path.exists(fmu_path):
        os.remove(fmu_path)
    with zipfile.ZipFile(fmu_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, content in paths.items():
            z.writestr(rel, content)
    write_progress(wd, "zip", 25.0)

    # 3) zip 结构校验: unzip -t + zipfile 完整性 ----------------------------
    up = subprocess.run(["unzip", "-t", fmu_path], capture_output=True,
                        text=True)
    unzip_t_ok = up.returncode == 0 and "No errors detected" in up.stdout
    with zipfile.ZipFile(fmu_path) as z:
        bad = z.testzip()
        entries = z.namelist()
    zip_ok = unzip_t_ok and bad is None and sorted(entries) == sorted(paths)
    write_progress(wd, "zip_check", 35.0)

    # 4) XML 关键字段断言 ---------------------------------------------------
    xml_checks = check_xml(os.path.join(wd, "modelDescription.xml"), fs)
    write_progress(wd, "xml_check", 40.0)

    # 5) C89 数值核编译 (driver 与核同译, gcc -std=c89) ----------------------
    drv_c = core_c.replace("#ifdef ROM_CORE_DRIVER", "#define ROM_CORE_DRIVER 1\n"
                           "#ifdef ROM_CORE_DRIVER")
    drv_c = drv_c.replace("ROM_TRAJ_PATH", '"' + os.path.join(wd, "traj.csv") +
                          '"')
    with open(os.path.join(wd, "sources", "rom_driver.c"), "w",
              encoding="utf-8") as f:
        f.write(drv_c)
    bin_path = os.path.join(wd, "rom_driver")
    cc = subprocess.run(["gcc", "-std=c89", "-pedantic", "-Wall", "-Wextra",
                         "-O2", "-o", bin_path,
                         os.path.join(wd, "sources", "rom_driver.c")],
                        capture_output=True, text=True)
    if cc.returncode != 0:
        raise RuntimeError(f"C89 编译失败: {cc.stderr}")
    write_progress(wd, "compile", 50.0)

    # 6) 同工况验收: 额定阶跃 + 循环扫描 (与 M96 同 dt/stride) ---------------
    from drv_spec import foc_simulate_constant
    run_case.bin_path = bin_path
    run_case.traj_path = os.path.join(wd, "traj.csv")
    run_case.foc = foc_simulate_constant
    floors = {"iq_A": fs["fmu_iq_abs_floor_a"],
              "omega_rad_s": fs["fmu_w_abs_floor_rad_s"],
              "te_Nm": 1.5 * sp["p"] * sp["lam"] * fs["fmu_iq_abs_floor_a"]}
    col_map_m96 = {"iq_A": (3, 3, floors["iq_A"]),
                   "omega_rad_s": (1, 1, floors["omega_rad_s"]),
                   "te_Nm": (2, 2, floors["te_Nm"])}
    cases = []
    scan = next(c for c in drv["cycles"] if c["name"] == "speed_scan")
    rated = next(c for c in drv["cycles"] if c["name"] == "rated_constant")
    case_list = [("rated_constant", rated["speed_rpm"], rated["torque_Nm"],
                  rated["t_end_s"])]
    for w in scan["speeds_rpm"]:
        case_list.append((f"scan_{int(w)}rpm", w, scan["torque_Nm"],
                          scan["t_end_s"]))
    tol = fs[REL_ERR_TOL_KEY] / 100.0
    for i, (name, w_rpm, tl, t_end) in enumerate(case_list):
        c_rows, py_rows, m96_rows = run_case(w_rpm, tl, t_end, dt, stride,
                                             drv, sp)
        worst_c_py, at_c_py = max_rel_err(c_rows, py_rows, col_map_m96)
        worst_c_m96, at_c_m96 = max_rel_err(c_rows, m96_rows, col_map_m96)
        case_ok = all(v <= tol for v in worst_c_m96.values())
        cases.append({
            "case": name, "speed_rpm": w_rpm, "torque_Nm": tl, "t_end_s": t_end,
            "n_samples": len(c_rows),
            "max_rel_err_c_vs_pyref_pct": {k: round(v * 100.0, 6)
                                           for k, v in worst_c_py.items()},
            "max_rel_err_c_vs_m96_pct": {k: round(v * 100.0, 6)
                                         for k, v in worst_c_m96.items()},
            "max_rel_err_at_t_s_c_vs_m96": at_c_m96,
            "t_wound_end_C": c_rows[-1][4],
            "t_wound_pyref_end_C": py_rows[-1][4],
            "m96_t_wound": "无 (M96 热为单向稳态代理, 无动态热状态, 披露)",
            "pass": case_ok,
        })
        write_progress(wd, f"case:{name}",
                       50.0 + (i + 1) / len(case_list) * 45.0,
                       time.time() - t0)

    # 7) 汇总报告 -----------------------------------------------------------
    overall = (zip_ok and xml_checks["all_pass"]
               and all(c["pass"] for c in cases))
    report = {
        "module": "M97",
        "title": "ROM/FMU 导出 (FMI 2.0 Co-Simulation, 自足生成)",
        "constants_src": ["constants/fmu_design.json",
                          "constants/drive_design.json (drv_* 继承)"],
        "fmu_file": os.path.relpath(fmu_path, wd),
        "fmu_entries": entries,
        "fmi": {"version": fs["fmu_fmi_version"], "type": fs["fmu_fmi_type"],
                "modelIdentifier": fs["fmu_model_identifier"],
                "n_continuous_states": 3,
                "state_decomposition": "dq 机电 2 状态 (iq, ω) + 热合并节点 1 状态"
                                       " (T_wound); id 由 FOC d 轴环钳位消去"},
        "thermal_lptn_merged": thm,
        "zip_check": {"unzip_t_ok": unzip_t_ok, "testzip_bad_entry": bad,
                      "entries_match": zip_ok, "unzip_t_stdout": up.stdout.strip()},
        "xml_checks": xml_checks,
        "c89_build": {"compiler": "gcc -std=c89 -pedantic -Wall -Wextra -O2",
                      "warnings": cc.stderr.strip() or "(无)"},
        "verification": {
            "protocol": "C89 数值核 vs Python 参考 ROM vs M96 run_drive_cycle "
                        "foc_simulate_constant 全阶 3 状态参考 (同 dt/stride 同工况)",
            "rel_err_convention": "|a−ref|/max(|ref|, 绝对下限); 下限 "
                                  f"iq={floors['iq_A']}A, ω={floors['omega_rad_s']}"
                                  f"rad/s, Te={floors['te_Nm']}Nm (全披露)",
            "tol_pct": fs[REL_ERR_TOL_KEY], "dt_s": dt, "stride": stride,
            "cases": cases,
        },
        "overall_status": "PASS" if overall else "FAIL",
        "caveats": fs["caveats"] + [
            "M96 全阶参考含 id 状态与 d 轴 PI, ROM 消去; 两者轨迹差异即为降阶误差,"
            " 以 ≤0.5% 相对误差验收 (全披露)",
            "T_wound 无 M96 动态对应量, 只做 C 核 vs Python 参考一致性披露",
        ],
        "honest_disclosure": True,
        "elapsed_s": round(time.time() - t0, 2),
    }
    out = os.path.join(wd, "rom_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_progress(wd, "done", 100.0, time.time() - t0)
    print(f"M97 ROM/FMU 导出: overall={report['overall_status']}")
    for c in cases:
        errs = c["max_rel_err_c_vs_m96_pct"]
        print(f"  {c['case']}: max_rel_err(C vs M96)="
              f"{max(errs.values()):.4f}% (iq={errs['iq_A']}%, "
              f"ω={errs['omega_rad_s']}%, Te={errs['te_Nm']}%) "
              f"T_w_end={c['t_wound_end_C']}°C pass={c['pass']}")
    print(f"zip: unzip -t {'OK' if unzip_t_ok else 'FAIL'}, "
          f"xml all_pass={xml_checks['all_pass']}")
    print(f"FMU: {fmu_path}\n报告: {out}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

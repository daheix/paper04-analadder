#!/usr/bin/env python3
"""M92 概念设计解析正向流驱动器 — 对标 SPEED 概念流

用法:
    python3 run_concept_design.py                       # 默认目标: 10.5Nm@3000rpm, 220V
    python3 run_concept_design.py --torque 10.5 --speed-rpm 3000 \
        --voltage 220 --p 2 --workdir /tmp/cpt1
    python3 run_concept_design.py --calibrate           # 3 标定库抽样 vs FEM 全披露

口径: concept_spec.py (σ=E·A 切应力 D²L 定尺, 磁路/绕组/损耗粗估);
      解析近似, 全部误差诚实披露, 禁虚标。
输出: <workdir>/concept_design_report.json + _progress.json (进度直通)。
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from load_constants import cooling as _cooling_fn
from concept_spec import concept_design, bg_from_mag_circuit
from load_constants import default_design, materials

# 3 个标定库抽样模型 (reference_metrics 含 FEM Bg1_T)
CALIB_MODELS = ["baseline-12s4p", "jac314-halbach-spm", "prius-2010-48s8p"]
LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "models", "library")


def write_progress(workdir, stage, percent, eta_s=None):
    """_progress.json 进度直通 (C++ 1s 轮询口径)。"""
    with open(os.path.join(workdir, "_progress.json"), "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "percent": round(percent, 1),
                   "eta_s": eta_s, "ts": time.time()}, f, ensure_ascii=False)


def load_calib_model(model_id):
    """读标定库模型 (fail-fast)。"""
    path = os.path.join(LIB_DIR, f"{model_id}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def calibrate():
    """3 抽样模型: 磁路解析 Bg1 vs FEM 基波, 误差全披露 (禁虚标)。"""
    mat = materials()["magnet_N35UH"]
    rows = []
    for mid in CALIB_MODELS:
        m = load_calib_model(mid)
        d, rm = m["design"], m["reference_metrics"]
        hm, g = d["mag_t"], d["R_si"] - (d["R_ri"] + d["mag_t"])
        bg_flat = bg_from_mag_circuit(mat["Br_20"], hm, g, mat["mur"])
        # 波形系数 (constants concept_waveform_coeff, 默认 1.0 — 并入披露误差)
        bg1_est = bg_flat
        bg1_fem = rm["Bg1_T"]
        rows.append({
            "model_id": mid,
            "p": d["p"], "n_slots": d["n_slots"],
            "hm_mm": round(hm * 1e3, 2), "g_mm": round(g * 1e3, 2),
            "bg1_analytic_T": round(bg1_est, 4),
            "bg1_fem_T": bg1_fem,
            "err_pct": round((bg1_est - bg1_fem) / bg1_fem * 100.0, 2),
        })
    max_err = max(abs(r["err_pct"]) for r in rows)
    return {
        "method": "磁路直算 Br·hm/(hm+μr·g), 忽略槽齿谐波/漏磁/轭饱和 (概念流解析口径)",
        "samples": rows, "max_abs_err_pct": max_err,
        "honest_disclosure": True,
        "note": "解析近似给量级; FEM 精化走 run_emag_getdp.py 全链",
    }


def main():
    ap = argparse.ArgumentParser(description="M92 概念设计解析正向流 (对标 SPEED)")
    ap.add_argument("--torque", type=float, default=10.5, help="目标转矩 Nm")
    ap.add_argument("--speed-rpm", type=float, default=3000.0)
    ap.add_argument("--voltage", type=float, default=220.0)
    ap.add_argument("--p", type=int, default=2, help="极对数")
    ap.add_argument("--bg1", type=float, default=None, help="磁负荷覆盖")
    ap.add_argument("--calibrate", action="store_true", help="附 3 抽样 FEM 误差披露")
    ap.add_argument("--workdir", default=HERE, help="工作目录隔离 (默认原地)")
    args = ap.parse_args()

    wd = args.workdir
    os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    write_progress(wd, "concept_sizing", 10.0, 2)
    rep = concept_design(target_torque=args.torque, speed_rpm=args.speed_rpm,
                         voltage=args.voltage, p=args.p, bg1_target=args.bg1)
    write_progress(wd, "calibration", 60.0, 1)
    rep["calibration"] = calibrate() if args.calibrate else None
    rep["cooling_reference"] = {
        "topologies": _cooling_fn()["cooling_topologies"],
        "note": "概念流温度为对流面积折算粗估; LPTN/FEM 热链见 run_thermal_getdp.py",
    }
    rep["elapsed_s"] = round(time.time() - t0, 3)
    write_progress(wd, "done", 100.0, 0)
    out = os.path.join(wd, "concept_design_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    s, e = rep["sizing"], rep["estimates"]
    print(f"== M92 概念设计正向流 (T={args.torque}Nm, n={args.speed_rpm:.0f}rpm, "
          f"U={args.voltage}V, p={args.p}) ==")
    print(f"  定尺: σ={s['sigma_Pa']}Pa → D={s['D_mm']}mm, L={s['L_mm']}mm, "
          f"τ_p={s['tau_p_mm']}mm")
    print(f"  绕组: N_ph={e['N_ph']} (float {e['N_ph_float']}), "
          f"λ_m={e['lam_m_Wb']:.4f}Wb, E_back={e['E_back_V']}V")
    print(f"  估算: I_q={e['I_q_A']}A, R_ph={e['R_ph_20C_ohm']}Ω, "
          f"P_cu={e['P_cu_W']}W, P_fe={e['P_fe_W']}W, η≈{e['eta_pct']}%, "
          f"ΔT≈{e['dT_winding_K_rough']}K (粗估)")
    cal = rep["calibration"]
    if cal:
        for r in cal["samples"]:
            print(f"  校准 {r['model_id']}: 解析 {r['bg1_analytic_T']}T vs FEM "
                  f"{r['bg1_fem_T']}T → err {r['err_pct']:+.2f}%")
        print(f"  最大误差 {cal['max_abs_err_pct']}% (解析近似, 全披露)")
    print(f"→ {out}")
    return rep


if __name__ == "__main__":
    main()

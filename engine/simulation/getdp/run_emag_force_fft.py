#!/usr/bin/env python3
"""M89 气隙力密度 FFT 链驱动器 — 径向力密度 σr=Bn²/2μ0 时空序列 → 2D 力谱

链路: generate_geo(Δ=0) → motor_mag_t.pro 单步 TimeLoopTheta 相位扫掠 K 步
      (M86 瞬态链扩展: σ=0 磁准静态口径, 磁场与电流瞬时值一一对应, 单步相位
      窗口 ≡ 瞬态链逐步解; 证据 /tmp/m89_probe 变体1, VT≈3064/步)
      → gap_circle 采样圆 (r_gap_mid, 与 M83 扭矩/空载同一几何)
      → bin_profile 空间均匀分 bin → force_fft_spec.spectrum_2d
      → force_fft_report.json (order×mode 矩阵 + 主阶次标注)

进度: _progress.json (ProgressReporter, 与静磁/瞬态链同 schema, C++ 1s 轮询)。
口径卡: force_fft_spec.py (doctest 可执行, constants/ force_fft_* 单一事实源)。
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_emag_getdp as base   # noqa: E402
from run_emag_getdp import GetDPMotorDriver, ProgressReporter  # noqa: E402
from run_emag_transient import GetDPTransientDriver  # noqa: E402
from transient_spec import load_transient_spec  # noqa: E402
from load_constants import operating  # noqa: E402
from run_emag_transient import electrical_freq_hz, time_step_s  # noqa: E402
import force_fft_spec as fs  # noqa: E402


def maxwell_torque_circle(theta_deg, bn, bt, r_sample, l_stack):
    """采样圆 Maxwell 转矩 T = L·r²·∮BnBt/μ0 dθ (与 GetDPMotorDriver.torque 同口径)。"""
    p_tan = bn * bt / fs.MU0
    return float(l_stack * r_sample ** 2 * p_tan.mean() * 2 * math.pi)


def main():
    overrides = None
    if "--params" in sys.argv:
        overrides = json.load(open(sys.argv[sys.argv.index("--params") + 1],
                                   encoding="utf-8"))
        overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
        base.P = dict(base.P, **merged_design(overrides))
        print(f"[params] UI参数覆盖: {sorted(overrides.keys())}")
    wd = None
    if "--workdir" in sys.argv:
        wd = sys.argv[sys.argv.index("--workdir") + 1]
        os.makedirs(wd, exist_ok=True)

    fspec = fs.load_force_fft_spec()          # constants/ force_fft_*
    tspec = load_transient_spec()             # constants/ transient_*
    oc = operating()
    if "--steps-per-period" in sys.argv:
        tspec["steps_per_period"] = int(sys.argv[sys.argv.index("--steps-per-period") + 1])
    js_pk = float(oc["effmap_rated_js"])
    if "--js-peak" in sys.argv:
        js_pk = float(sys.argv[sys.argv.index("--js-peak") + 1])

    n_poles = int(base.P["n_poles"])
    fe = electrical_freq_hz(float(oc["speed_rpm"]), n_poles)
    K = int(tspec["steps_per_period"])
    dt = time_step_s(1.0 / fe, K)
    r_sample = base.P["r_gap_mid"]
    l_stack = float(base.P["L_stack"])

    drv = GetDPTransientDriver(workdir=wd) if wd else GetDPTransientDriver()
    prog = ProgressReporter(drv.wd, run_tag="emag_force_fft")
    drv.prog = prog
    t_start = time.time()
    rep = {
        "params": {
            "speed_rpm": oc["speed_rpm"], "f_el_hz": fe, "n_poles": n_poles,
            "js_peak_Am2": js_pk, "steps_per_period": K, "dt_s": dt,
            "r_sample_m": r_sample, "mu0": fs.MU0,
            "sigma_r_formula": "Bn^2/(2*mu0)", "sigma_unit": "Pa",
            "theta_bins": fspec["theta_bins"], "max_mode": fspec["max_mode"],
            "top_n": fspec["top_n"],
            "method": "single-step TimeLoopTheta phase sweep (M86 window)",
        },
        "spectrum": {}, "dominant": [], "check": {},
        "runtime_s": 0.0, "log": [],
    }

    # ---- 几何/网格 (Δ=0, 与静磁/瞬态链同网格) ----
    prog.step("几何/网格", 3, "generate_geo(Δ=0) → gmsh")
    drv.mesh_rotor(0.0)

    # ---- 静磁载流基线 (转矩对标 + 时间均值 Bn 披露) ----
    prog.step("静磁载流基线", 8, f"δ=90° js_pk={js_pk:.3g} A/m² 非线性")
    ia = base.js_to_i(js_pk)
    drv.solve(ia=ia, ib=-ia / 2, ic=-ia / 2, linear=False,
              out_bmap="b_map_static_load.pos")
    th_s, bn_s, bt_s = drv.gap_circle("b_map_static_load.pos")
    t_static = maxwell_torque_circle(th_s, bn_s, bt_s, r_sample, l_stack)
    rep["static_baseline"] = {
        "torque_Nm": round(t_static, 4),
        "bn_max_T": round(float(np.abs(bn_s).max()), 4),
    }

    # ---- 相位扫掠: K 个单步瞬态窗口, 每步采样气隙圆 ----
    prog.step("相位扫掠", 15, f"{K} 步/电周期 × 单步窗口瞬态")
    prof = np.zeros((K, fspec["theta_bins"]))
    torques = np.zeros(K)
    for k in range(1, K + 1):
        t1 = round(k * dt, 12)
        f = f"bgap_k{k:03d}.pos"
        drv.solve_transient_window(t1 - dt, t1, js_pk, tspec, fe, out_bmap=f)
        th, bn, bt = drv.gap_circle(f)
        prof[k - 1] = fs.bin_profile(th, bn, fspec["theta_bins"])
        torques[k - 1] = maxwell_torque_circle(th, bn, bt, r_sample, l_stack)
        prog.step("相位扫掠", 15.0 + 70.0 * k / K,
                  f"第 {k}/{K} 步 φ={360.0 * k / K:.0f}° "
                  f"Bn_max={np.abs(bn).max():.3f}T T={torques[k - 1]:.3f}Nm")

    # ---- 2D 力谱 + 主阶次标注 ----
    prog.step("FFT 力谱", 88, f"spectrum_2d bins={fspec['theta_bins']} "
                              f"max_mode={fspec['max_mode']}")
    orders, modes, F = fs.spectrum_2d(prof, fspec["max_mode"])
    dom = fs.dominant(F, orders, modes, fspec["top_n"])

    # 验收: 主非 DC 峰 = 磁体基波×电枢反应交叉项 (h=1, m=0, 呼吸模态);
    #       σr=Bn² 的 DC 与空间 2p 谐波 (h=0,m=2p) 为静力背景, 单独披露;
    #       h=2 (Barm² 平方项) 弱 1 个量级以上, 列次峰。转矩对标: 扫掠均值 vs 静磁载流。
    t_mean = float(torques.mean())
    h0, m0 = dom[0]["h"], dom[0]["m"]
    h_ac, m_ac, a_ac = next((d["h"], d["m"], d["amp_Pa"])
                            for d in dom if d["h"] > 0.0)
    rep["spectrum"] = {
        "orders": orders, "modes": modes,
        "F_Pa": [[round(float(x), 3) for x in row] for row in F],
    }
    rep["dominant"] = dom
    rep["check"] = {
        "expected_peak": {"h": 1.0, "m": 0,
                          "why": "磁体基波×电枢反应交叉项 2·Bmag·Barm·cos(2πfe·t) "
                                 "→ h=1; 转子固定扫掠口径下 Bmag 空间 DC 分量主导 → m=0 (呼吸模态)"},
        "dominant_peak": {"h": h0, "m": m0, "amp_Pa": dom[0]["amp_Pa"]},
        "dominant_ac_peak": {"h": h_ac, "m": m_ac, "amp_Pa": a_ac},
        "peak_match_pass": bool(h_ac == 1.0 and m_ac == 0),
        "torque_sweep_mean_Nm": round(t_mean, 4),
        "torque_static_Nm": round(t_static, 4),
        "torque_err_pct": round(abs(t_mean - t_static) / abs(t_static) * 100.0, 2)
        if t_static != 0 else None,
        "sigma_r_dc_max_Pa": round(float(F[0].max()), 1),
        "sigma_r_m2p_dc_Pa": round(float(F[0][modes.index(2 * (n_poles // 2))]), 1),
        "sigma_r_peak_Pa": round(float(F.max()), 1),
        "bn_profile_phase_mean_max_T": round(float(np.abs(prof.mean(axis=0)).max()), 4),
    }

    rep["runtime_s"] = round(time.time() - t_start, 1)
    rep["log"] = drv.log
    out = os.path.join(drv.wd, "force_fft_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    prog.finish(True, f"M89 力谱: 主峰 h={h0} m={m0} "
                      f"{dom[0]['amp_Pa']:.0f}Pa 转矩对标 "
                      f"{rep['check']['torque_err_pct']}%")
    print(f"[M89] 完成: {out} ({rep['runtime_s']}s) "
          f"主峰 h={h0} m={m0} {dom[0]['amp_Pa']:.1f}Pa "
          f"转矩对标 {rep['check']['torque_err_pct']}%")


if __name__ == "__main__":
    main()

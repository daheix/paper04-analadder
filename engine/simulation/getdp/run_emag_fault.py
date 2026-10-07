#!/usr/bin/env python3
"""M88 短路/匝间故障工况链驱动壳 — 复用 M86 瞬态链, 禁业务数值逻辑

链路: generate_geo.py(Δ=0) → motor_mag_t.pro (TimeLoopTheta, M86 同口径)
      → 额定基线退磁 → sym_sc 三相对称短路 → turn_short 匝间短路
      → 每工况退磁膝点校核 (工作点 B_par vs B_knee) → fault_report.json

用法:
  python3 run_emag_fault.py [--params overrides.json] [--workdir DIR]
      [--skip-baseline]

口径来源 (单一事实源):
  - constants/operating_conditions.json: fault_* / transient_* / effmap_rated_js
  - fault_spec.py: 工况电流注入 / 退磁裕度 / 报告键序
  - transient_spec.py: θ 法 / dt / 电角频率 (M86 同口径)
  - 退磁判据复用 run_emag_getdp.demag_check (KNEE_PERMEANCE 膝点, JAC034)

验收: 每工况 risk_fraction == 0 (磁体工作点不低于膝点) 判 PASS。
进度: _progress.json (ProgressReporter, 与静磁/瞬态链同 schema)。
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_emag_getdp as base                     # noqa: E402
from run_emag_getdp import GetDPMotorDriver, ProgressReporter, operating  # noqa: E402
from run_emag_transient import GetDPTransientDriver                     # noqa: E402
from transient_spec import electrical_freq_hz, load_transient_spec       # noqa: E402
from fault_spec import (load_fault_spec, fault_cases, sc_periods,        # noqa: E402
                        demag_margin, report_keys_ok)


def main():
    overrides = None
    if "--params" in sys.argv:
        overrides = json.load(open(sys.argv[sys.argv.index("--params") + 1],
                                   encoding="utf-8"))
        overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
        base.P.update(merged_design_safe(overrides))
        print(f"[params] UI参数覆盖: {sorted(overrides.keys())}")
    wd = None
    if "--workdir" in sys.argv:
        wd = sys.argv[sys.argv.index("--workdir") + 1]
        os.makedirs(wd, exist_ok=True)
    skip_baseline = "--skip-baseline" in sys.argv

    fspec = load_fault_spec()                          # constants/ 单一事实源
    tspec = load_transient_spec()
    oc = operating()
    js_rated = float(oc["effmap_rated_js"])
    fe = electrical_freq_hz(float(oc["speed_rpm"]), int(base.P["n_poles"]))
    cases = fault_cases(js_rated, fspec)

    drv = GetDPTransientDriver(workdir=wd) if wd else GetDPTransientDriver()
    prog = ProgressReporter(drv.wd, run_tag="emag_fault")
    drv.prog = prog
    t0 = time.time()
    rep = {"params": {"f_el_hz": fe, "js_rated_Am2": js_rated, **fspec, **tspec,
                      "n_poles": int(base.P["n_poles"])},
           "cases": [], "log": []}

    # ---- 额定静磁基线退磁 (对比口径, 可跳过) ----
    prog.step("几何/网格", 3, "generate_geo(Δ=0) → gmsh")
    drv.mesh_rotor(0.0)
    if not skip_baseline:
        prog.step("额定基线", 12, f"静磁 load js={js_rated:.3g} 退磁基线")
        drv.solve(ia=js_rated, linear=False, out_bmap="b_map_fault_base.pos",
                  out_az="az_slots_fault_base.pos")
        dem0 = drv.demag_check("b_map_fault_base.pos")
        m0 = demag_margin(dem0)
        rep["baseline_rated"] = {"Bg1_T": drv.no_load_metrics(
            "b_map_fault_base.pos")["Bg1"], **dem0, **m0}
        print(f"=== 额定基线 ===  B_par_min={dem0['B_par_min_T']} T "
              f"knee={dem0['B_knee_T']} T margin={m0['margin_pct']}% "
              f"risk={dem0['risk_fraction']}")

    # ---- 两故障工况: M86 瞬态链注入 → 末步退磁校核 ----
    pcts = {"sym_sc": 30.0, "turn_short": 60.0}
    for name in ("sym_sc", "turn_short"):
        amps = cases[name]
        spec_k = dict(tspec, periods=sc_periods(fspec))
        prog.step("故障瞬态", pcts[name],
                  f"{name}: IA0={amps['IA0']:.3g} IB0={amps['IB0']:.3g} "
                  f"IC0={amps['IC0']:.3g} A/m² ({sc_periods(fspec)} 电周期)")
        # solve_transient 只收单一 js_pk (平衡三相), 匝间/短路需分相 → 直呼底层
        _solve_transient_amps(drv, amps, spec_k, fe,
                              f"b_map_fault_{name}.pos")
        nlk = drv.no_load_metrics(f"b_map_fault_{name}.pos")
        demk = drv.demag_check(f"b_map_fault_{name}.pos")
        mk = demag_margin(demk)
        rep["cases"].append({"name": name, "i_pk_Am2": amps["IA0"],
                             "periods": sc_periods(fspec),
                             "Bg1_T": nlk["Bg1"], "torque_Nm": drv.torque(
                                 f"b_map_fault_{name}.pos"),
                             **demk, **mk})
        print(f"=== {name} ===  Bg1={nlk['Bg1']} T  B_par_min={demk['B_par_min_T']} T "
              f"margin={mk['margin_pct']}% risk={demk['risk_fraction']} "
              f"{'PASS' if mk['pass'] else 'FAIL'}")

    # ---- 报告落盘 ----
    prog.step("报告落盘", 95, "fault_report.json")
    rep["runtime_s"] = round(time.time() - t0, 1)
    rep["log"] = drv.log
    assert report_keys_ok(rep)
    with open(os.path.join(drv.wd, "fault_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    ok = all(c["pass"] for c in rep["cases"])
    prog.finish(ok, f"故障工况 {'全部PASS' if ok else '存在退磁风险'} "
                    f"耗时{rep['runtime_s']}s → fault_report.json")
    sys.exit(0 if ok else 1)


def _solve_transient_amps(drv, amps, spec, fe, out_bmap):
    """分相幅值瞬态: 与 GetDPTransientDriver.solve_transient 同口径, 仅 IA0/IB0/IC0
    允许独立 (短路冲击/匝间叠加需要)。常数派生自 constants/, 零业务数值。"""
    import subprocess
    vals = {
        "TransT0": 0.0,
        "TransTMax": spec["periods"] * spec["steps_per_period"] *
                     (1.0 / fe / spec["steps_per_period"]),
        "TransDt": (1.0 / fe) / spec["steps_per_period"],
        "TransTheta": spec["theta"], "Fe": fe,
        "IA0": amps["IA0"], "IB0": amps["IB0"], "IC0": amps["IC0"],
        "SigMag": spec["sigma_magnet"], "SigIron": spec["sigma_iron"],
        "Flag_IronLinear": 0, "Mur_Iron": 7000, "Mur_Shaft": 1,
    }
    cmd = [base.GETDP_BIN, "motor_mag_t.pro", "-msh", "motor_sector.msh",
           "-solve", "Magnetodynamics_a"]
    for k, v in vals.items():
        cmd += ["-setnumber", k, repr(float(v))]
    t0 = time.time()
    drv.prog.check_stop()
    r = subprocess.run(cmd, cwd=drv.wd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"getdp 故障瞬态失败:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    drv.log.append(f"solve_fault({amps}): {time.time() - t0:.1f}s")
    if out_bmap != "b_map_t.pos":
        os.replace(os.path.join(drv.wd, "b_map_t.pos"),
                   os.path.join(drv.wd, out_bmap))


def merged_design_safe(overrides):
    from run_emag_getdp import merged_design
    return merged_design(overrides)


if __name__ == "__main__":
    main()

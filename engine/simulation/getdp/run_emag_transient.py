#!/usr/bin/env python3
"""M86 瞬态电磁链驱动壳 — 链路编排, 禁业务数值逻辑 (口径全部来自 constants/)

链路: generate_geo.py(rotor_angle_deg 剪切带步进) → gmsh → motor_mag_t.pro
      (GetDP TimeLoopTheta θ法) → b_map_t.pos → 与静磁 Bg1 对齐 → transient_report.json

用法:
  python3 run_emag_transient.py [--params overrides.json] [--workdir DIR]
      [--steps-per-period N] [--js-peak A/m2] [--rotor-steps N] [--skip-align]

口径来源 (单一事实源):
  - constants/operating_conditions.json: transient_theta / transient_periods /
    transient_steps_per_period / transient_sigma_magnet / transient_sigma_iron /
    transient_rotor_sweep_steps / speed_rpm / effmap_rated_js
  - transient_spec.py: 电角频率 / dt / 旋转步进约束 / Bg1 容差
  - 静磁基线复用 run_emag_getdp.GetDPMotorDriver (网格/解析/后处理逐位同口径)

验收: Δ=0 空载瞬态末步 Bg1 vs 静磁 Bg1 误差 ≤1% (σ=0 时矩阵逐位一致)。
进度: _progress.json (ProgressReporter, 与静磁链同 schema, C++ 1s 轮询)。
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_emag_getdp as base                     # noqa: E402
from run_emag_getdp import (GetDPMotorDriver,     # noqa: E402
                            ProgressReporter, merged_design, operating)
from transient_spec import (load_transient_spec,  # noqa: E402
                            electrical_freq_hz, time_step_s,
                            check_rotation_step, pole_pitch_deg, BGL_TOL)


class GetDPTransientDriver(GetDPMotorDriver):
    """瞬态驱动: 种子加 motor_mag_t.pro; 其余 (mesh/解析/扭矩) 复用静磁链。"""

    _SEED_FILES = GetDPMotorDriver._SEED_FILES + ("motor_mag_t.pro",)

    def solve_transient(self, js_pk, spec, fe, out_bmap="b_map_t.pos"):
        """一次 TimeLoopTheta 瞬态。js_pk: 相电流密度幅值 [A/m²] (与静磁同口径)。

        数值口径: TransT0=0 / TransTMax=periods*T_el / dt=T_el/steps /
        TransTheta / Fe / IA0..IC0=js_pk(平衡三相, 相位在 .pro 内) /
        SigMag/SigIron — 全部由 constants/ 与参数派生, 此处零业务常数。"""
        vals = {
            "TransT0": 0.0,
            "TransTMax": spec["periods"] * time_step_s(1.0 / fe, spec["steps_per_period"])
                       * spec["steps_per_period"],
            "TransDt": time_step_s(1.0 / fe, spec["steps_per_period"]),
            "TransTheta": spec["theta"],
            "Fe": fe,
            "IA0": js_pk, "IB0": js_pk, "IC0": js_pk,
            "SigMag": spec["sigma_magnet"], "SigIron": spec["sigma_iron"],
            "Flag_IronLinear": 0, "Mur_Iron": 7000, "Mur_Shaft": 1,
        }
        cmd = [base.GETDP_BIN, "motor_mag_t.pro", "-msh", "motor_sector.msh",
               "-solve", "Magnetodynamics_a"]
        for k, v in vals.items():
            cmd += ["-setnumber", k, repr(float(v))]
        t0 = time.time()
        self.prog.check_stop()
        import subprocess
        r = subprocess.run(cmd, cwd=self.wd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"getdp 瞬态失败:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        self.log.append(f"solve_transient(js={js_pk:.3g},fe={fe:.1f},"
                        f"steps={spec['steps_per_period']}): {time.time() - t0:.1f}s")
        if out_bmap != "b_map_t.pos":
            os.replace(os.path.join(self.wd, "b_map_t.pos"),
                       os.path.join(self.wd, out_bmap))
        return time.time() - t0

    def solve_transient_window(self, t0, t1, js_pk, spec, fe,
                               out_bmap="b_map_t.pos"):
        """单步窗口瞬态 [t0, t1] (M89 力密度相位扫掠用): 电流相位由 $Time 取值
        φ=2π·fe·t 给出; 每次调用落 bgap_t.pos (气隙 B 末步, motor_mag_t.pro)。
        其余数值口径与 solve_transient 完全一致 (constants/ 单一事实源)。"""
        vals = {
            "TransT0": t0, "TransTMax": t1,
            "TransDt": time_step_s(1.0 / fe, spec["steps_per_period"]),
            "TransTheta": spec["theta"],
            "Fe": fe,
            "IA0": js_pk, "IB0": js_pk, "IC0": js_pk,
            "SigMag": spec["sigma_magnet"], "SigIron": spec["sigma_iron"],
            "Flag_IronLinear": 0, "Mur_Iron": 7000, "Mur_Shaft": 1,
        }
        cmd = [base.GETDP_BIN, "motor_mag_t.pro", "-msh", "motor_sector.msh",
               "-solve", "Magnetodynamics_a"]
        for k, v in vals.items():
            cmd += ["-setnumber", k, repr(float(v))]
        t_start = time.time()
        self.prog.check_stop()
        import subprocess
        r = subprocess.run(cmd, cwd=self.wd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"getdp 单步窗口瞬态失败:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        self.log.append(f"solve_transient_window(t={t1:.6g}s,js={js_pk:.3g}): "
                        f"{time.time() - t_start:.1f}s")
        if out_bmap != "b_map_t.pos":
            os.replace(os.path.join(self.wd, "b_map_t.pos"),
                       os.path.join(self.wd, out_bmap))
        return time.time() - t_start

    def mesh_rotor(self, rotor_angle_deg):
        """剪切带步进网格: geo 参数注入 rotor_angle_deg (generate_geo 旋转转子环)。"""
        check_rotation_step(abs(rotor_angle_deg) % 360.0
                            if abs(rotor_angle_deg) <= 360.0 else 0.0, 360.0)
        base.P = dict(base.P, rotor_angle_deg=float(rotor_angle_deg))
        self.mesh()


def main():
    # ---- 参数: UI 覆盖 (--params) 与链开关 (--workdir 等), 无业务常数 ----
    overrides = None
    if "--params" in sys.argv:
        overrides = json.load(open(sys.argv[sys.argv.index("--params") + 1],
                                   encoding="utf-8"))
        overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
        base.P.update(merged_design(overrides))
        print(f"[params] UI参数覆盖: {sorted(overrides.keys())}")
    wd = None
    if "--workdir" in sys.argv:
        wd = sys.argv[sys.argv.index("--workdir") + 1]
        os.makedirs(wd, exist_ok=True)

    spec = load_transient_spec()                       # constants/ 单一事实源
    oc = operating()
    if "--steps-per-period" in sys.argv:
        spec["steps_per_period"] = int(sys.argv[sys.argv.index("--steps-per-period") + 1])
    js_pk = float(oc["effmap_rated_js"])
    if "--js-peak" in sys.argv:
        js_pk = float(sys.argv[sys.argv.index("--js-peak") + 1])
    n_rot = spec["rotor_sweep_steps"]
    if "--rotor-steps" in sys.argv:
        n_rot = int(sys.argv[sys.argv.index("--rotor-steps") + 1])
    skip_align = "--skip-align" in sys.argv

    fe = electrical_freq_hz(float(oc["speed_rpm"]), int(base.P["n_poles"]))
    dt = time_step_s(1.0 / fe, spec["steps_per_period"])

    drv = GetDPTransientDriver(workdir=wd) if wd else GetDPTransientDriver()
    prog = ProgressReporter(drv.wd, run_tag="emag_transient")
    drv.prog = prog
    t0 = time.time()
    rep = {"params": {"speed_rpm": oc["speed_rpm"], "f_el_hz": fe, "dt_s": dt,
                      "js_peak_Am2": js_pk, **spec,
                      "n_poles": int(base.P["n_poles"]),
                      "bg1_tol": BGL_TOL}, "log": []}

    # M100 温变磁体: 温度来源=LPTN 磁体节点 (T_Magnet_C, 单向); hc 向量段与
    # 静磁基线经 mesh()/solve() 的 magnet_temp_state 重算激励; 此处登记敏感度块
    _hc_t, _k, _a, _t = drv.magnet_temp_state(base.P)
    _ts = base.temp_spec.load_temp_spec()
    _dpct = base.temp_spec.sensitivity_pct(_a, _ts["sensitivity_dT_K"])
    rep["temp_sensitivity"] = dict(
        model=_ts["model"], disclosure=_ts["disclosure"],
        T_magnet_C=_t, alpha_br_per_K=_a, dT_K=_ts["sensitivity_dT_K"],
        bg1_change_pct=_dpct, torque_change_pct=_dpct, k_br=_k,
        Hc_mag_at_T_Am=_hc_t if _hc_t is not None else float(base.P["Hc0"]))

    # ---- 静磁基线 (与瞬态同一网格/同解析口径) ----
    prog.step("几何/网格", 3, "generate_geo(Δ=0) → gmsh")
    drv.mesh_rotor(0.0)
    pct0 = 10.0
    if not skip_align:
        prog.step("静磁基线", pct0, "motor_mag.pro 空载非线性")
        drv.solve(ia=0.0, linear=False, out_bmap="b_map_static_nl.pos",
                  out_az="az_slots_static.pos")
        nl_s = drv.no_load_metrics("b_map_static_nl.pos")
        rep["static_baseline"] = {"Bg1_T": nl_s["Bg1"],
                                  "Bg_pole_mean_T": nl_s["Bg_pole_mean"],
                                  "THD_pct": nl_s["THD_pct"],
                                  "flux_linkage_Wb": drv.flux_linkage(
                                      "az_slots_static.pos")["lambda_m_Wb"]}
        prog.step("静磁基线", pct0, f"Bg1={nl_s['Bg1']:.4f}T")
        pct0 = 25.0

    # ---- Δ=0 空载瞬态 (对齐基准: js=0 且 σ=0 时矩阵与静磁逐位一致) ----
    prog.step("瞬态推进", pct0, f"TimeLoopTheta θ={spec['theta']} "
                                f"{spec['steps_per_period']}步/周期 ×{spec['periods']}周期")
    drv.solve_transient(0.0, spec, fe)
    nl_t = drv.no_load_metrics("b_map_t.pos")
    t_fem = drv.torque("b_map_t.pos")
    rep["transient_d0"] = {"Bg1_T": nl_t["Bg1"], "THD_pct": nl_t["THD_pct"],
                           "torque_Nm": t_fem,
                           "flux_linkage_Wb": drv.flux_linkage(
                               "az_slots_t.pos")["lambda_m_Wb"]}
    if not skip_align:
        err = abs(nl_t["Bg1"] - nl_s["Bg1"]) / abs(nl_s["Bg1"])
        rep["alignment"] = {"bg1_static_T": nl_s["Bg1"], "bg1_transient_T": nl_t["Bg1"],
                            "err_pct": round(err * 100.0, 4),
                            "tol_pct": BGL_TOL * 100.0,
                            "pass": bool(err <= BGL_TOL)}

    # ---- 载流瞬态 (可选, 仅显式 --js-peak 时): 额定正弦三相 + 电枢反应 ----
    if "--js-peak" in sys.argv:
        prog.step("载流瞬态", 60.0, f"js_pk={js_pk:.3g} A/m² 平衡三相")
        drv.solve_transient(js_pk, spec, fe)
        nl_l = drv.no_load_metrics("b_map_t.pos")
        rep["transient_loaded"] = {
            "Bg1_T": nl_l["Bg1"], "THD_pct": nl_l["THD_pct"],
            "torque_Nm": drv.torque("b_map_t.pos"),
            "flux_linkage_Wb": drv.flux_linkage("az_slots_t.pos")["lambda_m_Wb"]}

    # ---- 转子剪切带步进扫掠 (可选): 每角度一次瞬态, 记录 Bg1/T ----
    if n_rot > 0:
        pitch = pole_pitch_deg(int(base.P["n_poles"]))   # 电周期=360/n_poles 机械度
        step_deg = pitch / n_rot
        rows = []
        for k in range(1, n_rot + 1):
            ang = round(k * step_deg, 6)
            check_rotation_step(step_deg, float(base.P["dtheta_max"]))
            prog.step("转子步进扫掠", pct0 + 60.0 * k / n_rot,
                      f"Δ={ang}° (第{k}/{n_rot}步)")
            drv.mesh_rotor(ang)
            drv.solve_transient(js_pk, spec, fe,
                                out_bmap=f"b_map_rot_{k:03d}.pos")
            nlk = drv.no_load_metrics(f"b_map_rot_{k:03d}.pos")
            rows.append({"rotor_angle_deg": ang, "Bg1_T": nlk["Bg1"],
                         "torque_Nm": drv.torque(f"b_map_rot_{k:03d}.pos")})
        rep["rotor_sweep"] = {"step_deg": step_deg, "rows": rows}

    # ---- 报告落盘 ----
    prog.step("报告落盘", 95, "transient_report.json")
    rep["runtime_s"] = round(time.time() - t0, 1)
    rep["log"] = drv.log
    with open(os.path.join(drv.wd, "transient_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    ok = skip_align or rep["alignment"]["pass"]
    prog.finish(ok, f"瞬态Bg1={nl_t['Bg1']:.4f}T "
                    f"(静磁{rep.get('static_baseline', {}).get('Bg1_T', float('nan')):.4f}T)"
                    f" 耗时{rep['runtime_s']}s → transient_report.json")
    if "alignment" in rep:
        a = rep["alignment"]
        print(f"=== M86 对齐 ===  静磁 {a['bg1_static_T']:.4f} T vs "
              f"瞬态 {a['bg1_transient_T']:.4f} T → 误差 {a['err_pct']}% "
              f"(容差 {a['tol_pct']}%) {'PASS' if a['pass'] else 'FAIL'}")
    print(f"=== Δ=0 瞬态 ===  Bg1={nl_t['Bg1']:.4f}T THD={nl_t['THD_pct']:.1f}% "
          f"T={t_fem:+.4f}Nm  (fe={fe:.1f}Hz dt={dt*1e3:.2f}ms "
          f"{spec['steps_per_period']}步/周期)")
    print(f"总耗时 {rep['runtime_s']}s → transient_report.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M87 场路耦合 (电压源驱动) 链路编排 — 链路编排, 禁业务数值逻辑 (口径全部来自 constants/)

链路: generate_geo.py(Δ=0) → motor_mag.pro(静磁 Picard, 端口电流逐点场解)
      + python 相变量电压方程 (磁链迭代分段耦合, 口径卡 coupling_spec.py)
      → coupled2_report.json

用法:
  python3 run_emag_coupled.py [--params overrides.json] [--workdir DIR]
      [--steps-per-period N]

口径来源 (单一事实源):
  - constants/operating_conditions.json: coupling_itol / coupling_iter_max /
    coupling_relax / R_ph_ohm(冷态) / effmap_rated_js / transient_steps_per_period
  - coupling_spec.py: 电压方程离散 / 磁链迭代 / 收敛判据 / TORQUE_TOL=1%
  - 工作点锚: coupled_report.json coupled_final (热耦合终值 Hc/R_ph/记录 T_dl)
  - 静磁场解复用 run_emag_getdp.GetDPMotorDriver (网格/解析/扭矩逐位同口径)

验收: 电压驱动稳态在 ia=+i_pk 快照步的 Maxwell 转矩 vs 同网格静磁 load 锚
误差 ≤1%; 存储基线 T_dl=−0.8966 Nm 偏差单独披露 (M85 旧网格漂移)。
进度: _progress.json (ProgressReporter, 与静磁链同 schema, C++ 1s 轮询)。
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_emag_getdp as base                     # noqa: E402
from run_emag_getdp import (GetDPMotorDriver,     # noqa: E402
                            ProgressReporter, merged_design, operating,
                            parse_az)
from transient_spec import (time_step_s,          # noqa: E402
                            electrical_freq_hz)
from coupling_spec import (load_coupling_spec, TORQUE_TOL,   # noqa: E402
                           phase_slot_angles, js_from_i, i_from_js,
                           circuit_residual, check_converged,
                           estimate_l_acc, torque_err)


class GetDPCoupledDriver(GetDPMotorDriver):
    """场路耦合驱动: 磁场方程走静磁链 (端口电流逐点场解), 电压方程在本类。"""

    def flux_phases(self, az_path="az_slots.pos"):
        """分相磁链 λ_ph = 2·Nc·L·[Σ A(+带槽心) − Σ A(−带槽心)] [Wb]。

        槽中心带 Az 采样与 flux_linkage 同机制 (bin 均值 + 环形补洞),
        槽心角由 coupling_spec.phase_slot_angles (PHASE_BELT 唯一事实源)。"""
        cx, cy, az = parse_az(os.path.join(self.wd, az_path))
        r = np.hypot(cx, cy)
        th = np.radians(np.degrees(np.arctan2(cy, cx)) % 360.0)
        band = (base.P["R_si"] + 0.5 * base.P["slot_depth"] - 1e-3 < r) \
             & (r < base.P["R_si"] + 0.5 * base.P["slot_depth"] + 1e-3)
        assert band.sum() > 50, f"槽中部 az 单元过少: {band.sum()}"
        nb = 720
        acc = np.zeros(nb); cnt = np.zeros(nb)
        ii = (th[band] / (2 * np.pi) * nb).astype(int) % nb
        np.add.at(acc, ii, az[band])
        np.add.at(cnt, ii, 1)
        m = cnt > 0
        prof = np.zeros(nb)
        prof[m] = acc[m] / cnt[m]
        idxs = np.arange(nb)
        prof[~m] = np.interp(idxs[~m], idxs[m], prof[m], period=nb)

        def at(deg):
            return float(prof[int(round(deg / 360.0 * nb)) % nb])

        nc, ls = float(base.P["Nc"]), float(base.P["L_stack"])
        out = {}
        for ph, (plus, minus) in phase_slot_angles(
                int(base.P["n_slots"])).items():
            s = sum(at(a) for a in plus) - sum(at(a) for a in minus)
            out[ph] = 2.0 * nc * ls * s
        return out


def load_anchor():
    """工作点锚: coupled_report.json coupled_final (热耦合终值)。

    >>> a = load_anchor()
    >>> a["hc"] > 8e5 and a["r_ph"] > 30.0 and a["t_dl_stored"] < 0
    True
    """
    with open(os.path.join(HERE, "coupled_report.json"), encoding="utf-8") as f:
        fin = json.load(f)["coupled_final"]
    return {"hc": float(fin["Hc_Am"]), "r_ph": float(fin["R_ph_ohm"]),
            "t_dl_stored": float(fin["T_dl_Nm"]),
            "t_w": float(fin["T_winding"]), "t_m": float(fin["T_magnet"])}


def js_waveform(k, n_steps, js_pk):
    """基线三相电流密度波 (M86 口径): A 相 sin, B −120°, C +120°。

    >>> js = js_waveform(10, 40, 2.0e6)
    >>> round(js["a"], 3), round(js["b"] / js["a"], 4), round(js["c"] / js["a"], 4)
    (2000000.0, -0.5, -0.5)
    """
    wt = 2.0 * math.pi * k / n_steps
    return {"a": js_pk * math.sin(wt),
            "b": js_pk * math.sin(wt - 2.0 * math.pi / 3.0),
            "c": js_pk * math.sin(wt + 2.0 * math.pi / 3.0)}


def main():
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

    spec = load_coupling_spec()                        # constants/ 单一事实源
    oc = operating()
    n_steps = int(oc["transient_steps_per_period"])
    if "--steps-per-period" in sys.argv:
        n_steps = int(sys.argv[sys.argv.index("--steps-per-period") + 1])
    js_pk = float(oc["effmap_rated_js"])
    r20 = float(oc["R_ph_ohm"])

    anchor = load_anchor()
    i_pk = i_from_js(js_pk, float(base.P["Nc"]), _slot_area())
    fe = electrical_freq_hz(float(oc["speed_rpm"]), int(base.P["n_poles"]))
    dt = time_step_s(1.0 / fe, n_steps)
    k_snap = n_steps // 4                              # ia=+i_pk 快照 (ωt=90°)

    drv = GetDPCoupledDriver(workdir=wd) if wd else GetDPCoupledDriver()
    prog = ProgressReporter(drv.wd, run_tag="emag_coupled")
    drv.prog = prog
    t0 = time.time()
    rep = {"params": {"speed_rpm": oc["speed_rpm"], "f_el_hz": fe, "dt_s": dt,
                      "js_peak_Am2": js_pk, "i_pk_A": i_pk, "n_steps": n_steps,
                      "r_ph_hot_ohm": anchor["r_ph"], "hc_hot_Am": anchor["hc"],
                      "t_dl_stored_Nm": anchor["t_dl_stored"],
                      "itol": spec["itol"], "iter_max": spec["iter_max"],
                      "relax": spec["relax"], "torque_tol": TORQUE_TOL},
           "anchor": anchor, "log": []}

    prog.step("几何/网格", 3, "generate_geo(Δ=0) → gmsh")
    drv.mesh()

    # ---- 标定波 (等效电压反演): 基线电流逐点场解 → λ^base_k + 静磁 load 锚 ----
    nc = float(base.P["Nc"])
    lam_base, i_base, t_anchor = [], [], None
    for k in range(n_steps):
        prog.step("标定波场解", 5.0 + 45.0 * k / n_steps,
                  f"基线电流场解 k={k + 1}/{n_steps}")
        jk = js_waveform(k, n_steps, js_pk)
        drv.solve(ia=jk["a"], ib=jk["b"], ic=jk["c"], linear=False,
                  clean_res=True, out_bmap="b_map_cal.pos",
                  hcmag=anchor["hc"])
        lam_base.append(drv.flux_phases())
        i_base.append({ph: i_from_js(jk[ph], nc, _slot_area())
                       for ph in "abc"})
        if k == k_snap:
            t_anchor = drv.torque("b_map_cal.pos")
    rep["anchor"]["t_dl_fresh_Nm"] = round(t_anchor, 6)

    # ---- 电压源标定 (周期延拓) + 加速电感 (弦斜率 ×1.5) ----
    u_src = []
    for k in range(n_steps):
        kp = (k - 1) % n_steps
        u_src.append({ph: anchor["r_ph"] * i_base[k][ph]
                      + (lam_base[k][ph] - lam_base[kp][ph]) / dt
                      for ph in "abc"})
    l_cal = []
    for ph in "abc":
        sl = [(lam_base[k][ph] - lam_base[k - 1][ph]) / (i_base[k][ph] - i_base[k - 1][ph])
              for k in range(n_steps)
              if abs(i_base[k][ph] - i_base[k - 1][ph]) > 1e-9]
        l_cal.append(float(np.median(sl)))
    l_acc = 1.5 * max(abs(x) for x in l_cal)
    rep["circuit"] = {"l_cal_H": [round(x, 6) for x in l_cal],
                      "l_acc_H": round(l_acc, 6),
                      "u_pk_V": {ph: round(max(abs(u[ph]) for u in u_src), 3)
                                 for ph in "abc"}}

    # ---- 电压驱动耦合 (磁链迭代分段耦合, 预报子=上一步收敛电流) ----
    i_cpl = dict(i_base[-1])                           # 周期首步: 末步解
    lam_prev = dict(lam_base[-1])
    rows, snap_bmap_saved, all_ok = [], False, True
    it_hist = []
    i_snap = None                                      # 快照步收敛电流 (报告用)
    for k in range(n_steps):
        prog.step("电压驱动耦合", 50.0 + 45.0 * k / n_steps,
                  f"耦合步 k={k + 1}/{n_steps}")
        i_prev = dict(i_cpl)
        i_try = dict(i_prev)
        it, conv = 0, False
        for it in range(1, spec["iter_max"] + 1):
            jt = {ph: js_from_i(i_try[ph], nc, _slot_area()) for ph in "abc"}
            drv.solve(ia=jt["a"], ib=jt["b"], ic=jt["c"], linear=False,
                      clean_res=True, out_bmap="b_map_cpl.pos",
                      hcmag=anchor["hc"])
            lam_try = drv.flux_phases()
            i_new = {}
            for ph in "abc":
                res = circuit_residual(u_src[k][ph], anchor["r_ph"],
                                       i_try[ph], lam_try[ph],
                                       lam_prev[ph], dt)
                i_new[ph] = i_try[ph] + spec["relax"] * res / (
                    anchor["r_ph"] + l_acc / dt)
            di = [i_new[ph] - i_try[ph] for ph in "abc"]
            i_try = dict(i_new)
            if check_converged(di, i_pk, spec["itol"]):
                conv = True
                break
        it_hist.append(it)
        all_ok = all_ok and conv
        if k == k_snap:
            os.replace(os.path.join(drv.wd, "b_map_cpl.pos"),
                       os.path.join(drv.wd, "b_map_cpl_snap.pos"))
            snap_bmap_saved = True
            i_snap = dict(i_try)
        lam_prev = dict(lam_try)
        i_cpl = dict(i_try)
        rows.append({"k": k, "iters": it, "converged": conv,
                     "i_A": {ph: round(i_cpl[ph], 6) for ph in "abc"},
                     "u_V": {ph: round(u_src[k][ph], 4) for ph in "abc"},
                     "lam_Wb": {ph: round(lam_prev[ph], 6) for ph in "abc"}})

    # ---- 验收: 快照步转矩对齐 ----
    prog.step("验收对齐", 96, "快照步转矩 vs 静磁 load 锚")
    t_cpl = drv.torque("b_map_cpl_snap.pos")
    err_fresh = torque_err(t_cpl, t_anchor)
    err_stored = torque_err(t_cpl, anchor["t_dl_stored"])
    rep["coupled"] = {
        "iters_total": int(np.sum(it_hist)),
        "iters_max": int(np.max(it_hist)),
        "all_converged": bool(all_ok),
        "torque_snap_Nm": round(t_cpl, 6),
        "i_snap_A": {ph: round(i_snap[ph], 6) for ph in "abc"},
        "alignment": {
            "t_anchor_fresh_Nm": round(t_anchor, 6),
            "err_vs_fresh_pct": round(err_fresh * 100.0, 4),
            "t_dl_stored_Nm": anchor["t_dl_stored"],
            "err_vs_stored_pct": round(err_stored * 100.0, 4),
            "tol_pct": TORQUE_TOL * 100.0,
            "pass": bool(err_fresh <= TORQUE_TOL)}}
    rep["steps"] = rows

    prog.step("报告落盘", 98, "coupled2_report.json")
    rep["runtime_s"] = round(time.time() - t0, 1)
    rep["log"] = drv.log
    with open(os.path.join(drv.wd, "coupled2_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    ok = all_ok and rep["coupled"]["alignment"]["pass"]
    a = rep["coupled"]["alignment"]
    prog.finish(ok, f"场路耦合 T_snap={t_cpl:.4f}Nm (静磁锚{a['t_anchor_fresh_Nm']:.4f} "
                    f"/ 记录基线{a['t_dl_stored_Nm']:.4f}) 耗时{rep['runtime_s']}s"
                    f" → coupled2_report.json")
    print(f"=== M87 场路耦合 ===  快照步 T={t_cpl:+.4f} Nm vs "
          f"同网格静磁锚 {a['t_anchor_fresh_Nm']:+.4f} Nm → 误差 "
          f"{a['err_vs_fresh_pct']}% (容差 {a['tol_pct']}%) "
          f"{'PASS' if a['pass'] else 'FAIL'}")
    print(f"    vs 记录基线 T_dl={a['t_dl_stored_Nm']:+.4f} Nm → 偏差 "
          f"{a['err_vs_stored_pct']}% (M85 旧网格漂移, 口径卡披露)")
    print(f"=== 电路侧 ===  内层迭代 合计{rep['coupled']['iters_total']} "
          f"峰值{rep['coupled']['iters_max']} 全收敛={all_ok}; "
          f"L_acc={l_acc:.4f}H, u_pk={rep['circuit']['u_pk_V']}")
    print(f"总耗时 {rep['runtime_s']}s → coupled2_report.json")
    sys.exit(0 if ok else 1)


def _slot_area():
    """槽截面积 (run_emag_getdp.slot_area 转发, 避免驱动内重算口径)。"""
    return base.slot_area()


if __name__ == "__main__":
    main()

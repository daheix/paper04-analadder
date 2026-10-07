#!/usr/bin/env python3
"""M96 驱动工况循环时域仿真驱动器 — dq+FOC+机械 ODE (对标 Motor-CAD Lab / JMAG-RT / PLECS)

用法:
    python3 run_drive_cycle.py                                  # 全部循环, 默认 dt
    python3 run_drive_cycle.py --workdir /tmp/m96a --dt 1e-4    # workdir 隔离+步长白名单
    python3 run_drive_cycle.py --cycle rated_constant            # 单循环
    python3 run_drive_cycle.py --integrator euler                # 步进欧拉 (默认 rk4)

口径: drv_spec.py (dq 电压方程/Te/J·dω/dt/FOC 双环/SVPWM Udc/√3 限幅);
      循环表 constants/drive_design.json drv_cycles (constant/step/scan);
      热联动单向披露: 铜损/铁损均值 → LPTN 温升代理 ΔT=P/(h·S), 无热反馈。
输出: <workdir>/drivecycle_report.json + _progress.json (进度直通)。
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from drv_spec import (em_torque, foc_simulate_constant, iq_from_torque,
                      load_drv, p_cu_dq)
from load_constants import cooling, default_design, drive_cycle, map_design, \
    materials, operating
from concept_spec import temp_rise_rough
from map_spec import eta_of, voltage_ok

STANDARD_CYCLES = ("wltc", "cltc")


def write_progress(workdir, stage, percent, eta_s=None):
    """_progress.json 进度直通 (C++ 1s 轮询口径)。"""
    with open(os.path.join(workdir, "_progress.json"), "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "percent": round(percent, 1),
                   "eta_s": eta_s, "ts": time.time()}, f, ensure_ascii=False)


def profile_constant(cycle):
    """恒速恒载: t → (w_ref_rpm, TL)。"""
    w = float(cycle["speed_rpm"])
    tl = float(cycle["torque_Nm"])
    return lambda t: (w, tl)


def profile_step(cycle):
    """阶跃起停: 分段常值 (t0,t1 区间查表, 区间断点右连续, 区间外 fail-fast)。"""
    segs = [(float(s["t0_s"]), float(s["t1_s"]), float(s["speed_rpm"]),
             float(s["torque_Nm"])) for s in cycle["segments"]]

    def f(t):
        for t0, t1, w, tl in segs:
            if t0 <= t < t1 or (t == segs[-1][1] and t0 <= t <= t1):
                return w, tl
        raise ValueError(f"step 循环时刻越界: t={t}")
    return f


def run_step_cycle(cycle, dt, sp, integrator):
    """step 循环: 分段顺序调用 foc_simulate_constant (状态不跨段续接, 披露)。"""
    segs = cycle["segments"]
    traj = []
    eta_num = eta_den = 0.0
    p_loss_sum = p_loss_w = 0.0
    v_max = i_max = 0.0
    max_scale = 1.0
    diverged = False
    t0 = time.time()
    for i, seg in enumerate(segs):
        sub = {"speed_rpm": seg["speed_rpm"], "torque_Nm": seg["torque_Nm"],
               "t_end_s": seg["t1_s"] - seg["t0_s"]}
        r = run_constant_cycle(sub, dt, sp, integrator)
        diverged = diverged or r["diverged"]
        for row in r["traj"]:
            traj.append([round(row[0] + seg["t0_s"], 9)] + row[1:])
        eta_num += r["e_out_J"]
        eta_den += r["e_in_J"] + r["e_fe_J"]
        p_loss_sum += r["p_loss_avg_W"] * sub["t_end_s"]
        v_max = max(v_max, r["bus_v_max_V"])
        i_max = max(i_max, r["bus_i_max_A"])
        max_scale = max(max_scale, r["max_scale"])
        write_progress_cwd("step_cycle", (i + 1) / len(segs) * 90.0, t0, len(segs))
    t_end = float(cycle["t_end_s"])
    return {"eta_pct": eta_num / eta_den * 100.0 if eta_den > 0 else 0.0,
            "bus_v_max_V": v_max, "bus_i_max_A": i_max, "max_scale": max_scale,
            "diverged": diverged, "traj": traj,
            "p_loss_avg_W": p_loss_sum / t_end, "t_end_s": t_end}


def run_constant_cycle(cycle, dt, sp, integrator):
    """单个恒工况: foc_simulate_constant + 母线量 + 稳态锚点核对。"""
    r = foc_simulate_constant(cycle["speed_rpm"], cycle["torque_Nm"],
                              cycle["t_end_s"], dt, sp, sample_stride=None,
                              integrator=integrator)
    p_in_end = 1.5 * (r["ud_end"] * r["id_end"] + r["uq_end"] * r["iq_end"])
    p_in_end = max(p_in_end, 0.0)
    r["bus_v_max_V"] = sp["Udc"]
    r["bus_i_max_A"] = p_in_end / sp["Udc"]
    if float(cycle["torque_Nm"]) > 0.0:
        iq_anchor = iq_from_torque(cycle["torque_Nm"], sp["p"], sp["lam"])
        r["iq_steady_check"] = {
            "iq_foc_A": round(r["iq_end"], 6),
            "iq_anchor_A": round(iq_anchor, 6),
            "dev_pct": round((r["iq_end"] - iq_anchor) / iq_anchor * 100.0, 3),
        }
    r["integrator"] = integrator
    # 稳态效率口径 (端点): eta_ss = TL·ω/(P_in+P_fe) — 剔除 0→n 起旋转子动能占比,
    # 用于与 M92 概念流 η峰 96.8% 口径对标; 时域综合 eta_pct 含起动过程 (如实披露)。
    w_end = r["w_end"]
    p_out_ss = float(cycle["torque_Nm"]) * w_end
    p_fe_ss = 0.0
    if w_end > 0.0:
        we_end = sp["p"] * w_end
        from drv_spec import p_fe_proxy
        p_fe_ss = p_fe_proxy(sp["pfe_hyst"], sp["pfe_eddy"],
                             abs(we_end) / (2.0 * math.pi), sp["pfe_fref"])
    den_ss = p_in_end + p_fe_ss
    r["eta_ss_pct"] = round(p_out_ss / den_ss * 100.0, 3) if den_ss > 0 else 0.0
    return r


def run_scan_cycle(cycle, dt, sp, integrator):
    """循环扫描表: 逐转速点独立恒工况仿真, 汇总综合效率。"""
    points = []
    e_out = e_in = 0.0
    for w_rpm in cycle["speeds_rpm"]:
        sub = {"speed_rpm": w_rpm, "torque_Nm": cycle["torque_Nm"],
               "t_end_s": cycle["t_end_s"]}
        r = run_constant_cycle(sub, dt, sp, integrator)
        points.append({"speed_rpm": w_rpm, "eta_pct": round(r["eta_pct"], 3),
                       "iq_steady_check": r.get("iq_steady_check"),
                       "bus_i_max_A": round(r["bus_i_max_A"], 4),
                       "p_loss_avg_W": r["p_loss_avg_W"],
                       "diverged": r["diverged"]})
        e_out += r["e_out_J"]
        e_in += r["e_in_J"] + r["e_fe_J"]
    return {"eta_pct": e_out / e_in * 100.0 if e_in > 0 else 0.0,
            "bus_v_max_V": sp["Udc"],
            "bus_i_max_A": max(p["bus_i_max_A"] for p in points),
            "max_scale": 1.0, "diverged": any(p["diverged"] for p in points),
            "traj": [], "scan_points": points,
            "p_loss_avg_W": (sum(p["p_loss_avg_W"] * cycle["t_end_s"]
                                 for p in points)
                             / (cycle["t_end_s"] * len(points))),
            "t_end_s": cycle["t_end_s"] * len(cycle["speeds_rpm"])}


def thermal_proxy(p_loss_avg_w, sp):
    """热联动 (单向披露): 损耗均值 → 温升代理 ΔT=P/(h·S)。

    h 取 cooling.json natural (自然冷回归口径), S=π·D_so·L_stack (默认设计)。
    仅驱动→热单向馈入, 无热反馈修改电参数 (披露)。
    """
    d = default_design()
    h = cooling()["cooling_topologies"]["natural"]["h_conv"]
    dia, length = 2.0 * d["R_so"], d["L_stack"]
    return {"p_loss_avg_W": round(p_loss_avg_w, 3),
            "h_conv_W_m2K": h, "D_so_m": dia, "L_stack_m": length,
            "dT_proxy_K": round(temp_rise_rough(p_loss_avg_w, h, dia, length), 2),
            "direction": "drive->thermal 单向 (无热反馈)", }


def standard_cycle_points(name, sp):
    """M98 标准循环 (wltc/cltc) 准静态工作点表: 1Hz 逐秒 (t, rpm, TL)。

    口径: 车速 km/h → 轮速 v; 加速度前向差分; 纵向力 F = m·a + A + C·v²;
    电机端 TL = F·r/(i_g·η_g) (驱动, η_g 折算; 制动段 TL<0 如实保留);
    rpm = v/(2πr)·60·i_g。整车常数 map_cycle_* (典型值假设, 披露见 map_caveats)。
    """
    raw = map_design()
    cyc = drive_cycle(name)
    m = raw["map_cycle_mass_kg"]
    r_w = raw["map_cycle_wheel_r_m"]
    ig = raw["map_cycle_gear_ratio"]
    eg = raw["map_cycle_gear_eff"]
    A, C = raw["map_cycle_road_a_N"], raw["map_cycle_road_c_Ns2_m2"]
    pts = cyc["speed_kmh"]
    dt = 1.0
    out = []
    n = len(pts)
    for i, kmh in enumerate(pts):
        v = kmh / 3.6
        a = ((pts[i + 1] - pts[i]) / 3.6 / dt) if i + 1 < n else 0.0
        f = m * a + A + C * v * v
        tl = f * r_w / (ig * eg)
        w_mech = v / r_w * ig
        out.append({"t_s": i, "v_kmh": kmh,
                    "speed_rpm": w_mech * 60.0 / (2.0 * math.pi),
                    "torque_Nm": tl})
    return {"name": name, "source": cyc.get("source"),
            "phases": cyc.get("phases"), "points": out, "dt_s": dt}


def run_standard_cycle(name, sp):
    """标准循环准静态评估: 逐秒 id=0 + iq_from_torque + 电压可行性 + M92 损耗效率。

    id=0 为保守控制口径 (非 MTPA, 效率偏保守, 披露); 超出电压椭圆/转速上限的秒数
    计 infeasible_s; 制动 (TL≤0) 能量单列 e_regen_available_J, 不折算再生回收率。
    """
    steel = materials()["steel_50JN350"]
    kh, ke = steel["KH"], steel["KE"]
    bg1 = map_design()["map_bg1_ref_T"]
    v_fe = operating()["V_fe_m3"]
    prof = standard_cycle_points(name, sp)
    dt = prof["dt_s"]
    phases = prof.get("phases") or {}
    e_in = e_out = e_regen = 0.0
    v_max = rpm_max = 0.0
    infeasible = braking = 0
    fail_reasons = {}
    per_phase = {}
    for p in prof["points"]:
        w_mech = p["speed_rpm"] * 2.0 * math.pi / 60.0
        rpm_max = max(rpm_max, p["speed_rpm"])
        v_max = max(v_max, p["v_kmh"])
        tl = p["torque_Nm"]
        if tl <= 0.0:
            braking += 1
            e_regen += -tl * w_mech * dt
            continue
        iq = iq_from_torque(tl, sp["p"], sp["lam"])
        id_v = 0.0
        we = sp["p"] * w_mech
        umax = sp["Udc"] / math.sqrt(3.0)
        # 解析弱磁: 电压椭圆超限时反解 id (保持 iq 转矩分量不变, id≥−60A 内)
        if we * math.hypot(sp["Ld"] * id_v + sp["lam"], sp["Lq"] * iq) > umax:
            r2 = (umax / we) ** 2 - (sp["Lq"] * iq) ** 2
            if r2 < 0.0:
                infeasible += 1
                fail_reasons["转矩需求超电压-电流能力 (r2<0)"] = \
                    fail_reasons.get("转矩需求超电压-电流能力 (r2<0)", 0) + 1
                continue
            id_v = (math.sqrt(r2) - sp["lam"]) / sp["Ld"]
        if p["speed_rpm"] > 10000.0 or abs(iq) > 60.0 or id_v < -60.0:
            infeasible += 1
            key = ("转速超上限" if p["speed_rpm"] > 10000.0
                   else "转矩需求超 iq=60A 电流上限")
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
            continue
        if not voltage_ok(id_v, iq, we, sp["Ld"], sp["Lq"], sp["lam"], sp["Udc"]):
            infeasible += 1
            fail_reasons["弱磁后仍超电压椭圆"] = \
                fail_reasons.get("弱磁后仍超电压椭圆", 0) + 1
            continue
        e_p_out = tl * w_mech * dt
        eta = eta_of(tl, w_mech, id_v, iq, sp["Rs"], kh, ke,
                     abs(we) / (2.0 * math.pi), bg1, v_fe) / 100.0
        e_out += e_p_out
        e_in += e_p_out / eta if eta > 0 else e_p_out
        for pname, rng in phases.items():
            if isinstance(rng, list) and rng[0] <= p["t_s"] <= rng[1]:
                ph = per_phase.setdefault(
                    pname, {"e_out_J": 0.0, "e_in_J": 0.0, "seconds": 0})
                ph["e_out_J"] += e_p_out
                ph["e_in_J"] += e_p_out / eta if eta > 0 else e_p_out
                ph["seconds"] += 1
                break
    for ph in per_phase.values():
        ph["eta_pct"] = round(ph["e_out_J"] / ph["e_in_J"] * 100.0, 3) \
            if ph["e_in_J"] > 0 else 0.0
        ph["e_out_MJ"] = round(ph.pop("e_out_J") / 1e6, 4)
        ph["e_in_MJ"] = round(ph.pop("e_in_J") / 1e6, 4)
    dist_km = sum(p["v_kmh"] for p in prof["points"]) / 3600.0
    return {"name": name, "kind": "standard", "source": prof["source"],
            "n_points": len(prof["points"]), "dt_s": dt,
            "v_max_kmh": round(v_max, 2), "rpm_max": round(rpm_max, 1),
            "dist_km": round(dist_km, 3),
            "eta_pct": round(e_out / e_in * 100.0, 3) if e_in > 0 else 0.0,
            "e_out_MJ": round(e_out / 1e6, 4), "e_in_mot_MJ": round(e_in / 1e6, 4),
            "e_regen_available_MJ": round(e_regen / 1e6, 4),
            "braking_s": braking, "infeasible_s": infeasible,
            "infeasible_reasons_s": fail_reasons,
            "per_phase": per_phase,
            "status": "PASS" if infeasible == 0 else "FAIL",
            "caveats": ["准静态逐秒工作点 (无瞬态/无起动动能), 解析弱磁反解 id",
                        "制动能量单列, 未折算再生回收率",
                        "整车常数 map_cycle_* 为典型假设值 (披露)",
                        "infeasible_s>0 表示本电机 (M96 额定 10.5Nm/iq≤60A) "
                        "与假设整车不匹配, 秒级不可行原因见 infeasible_reasons_s"]}


def m92_consistency(cycle, sp, eta_ss_pct=None):
    """与 M92 概念流口径一致性: id=0 转矩/铜损公式恒等 (0% 偏差披露);
    效率偏差如实披露 (驱动参数 Rs/iq 与 M92 概念流额定点不同属口径差异)。"""
    t = float(cycle["torque_Nm"])
    iq = iq_from_torque(t, sp["p"], sp["lam"])
    i_rms = iq / math.sqrt(2.0)
    p_cu_m92 = 3.0 * i_rms * i_rms * sp["Rs"]
    dev = (p_cu_dq(0.0, iq, sp["Rs"]) - p_cu_m92) / p_cu_m92 * 100.0
    out = {"torque_Nm": t, "iq_A": round(iq, 6),
           "p_cu_dq_W": round(p_cu_dq(0.0, iq, sp["Rs"]), 6),
           "p_cu_m92_W": round(p_cu_m92, 6), "dev_pct": round(dev, 9),
           "note": "dq 铜损 1.5·Rs·I² ≡ M92 的 3·I_rms²·R_ph (幅值约定恒等); "
                   "铁损为频率代理 vs M92 Steinmetz 基波, 差异见 drv_caveats"}
    if eta_ss_pct is not None:
        out["eta_ss_pct_drive"] = eta_ss_pct
        out["eta_m92_peak_pct"] = 96.8
        out["eta_dev_disclosure"] = (
            f"驱动链稳态效率 {eta_ss_pct}% vs M92 概念流 η峰 96.8%: 本卡参数 "
            f"Rs={sp['Rs']}Ω/iq≈{iq:.1f}A 下铜损占比高, 且 96.8% 为概念流峰值工况点, "
            "两者非同一定点, 偏差如实披露不做对齐")
    return out


def write_progress_cwd(stage, percent, t0, total):
    """过程进度 (runner 级 _progress.json, cwd 由 main 管理)。"""
    if hasattr(write_progress_cwd, "wd"):
        elapsed = time.time() - t0
        eta = elapsed / percent * (100.0 - percent) if percent > 1e-9 else None
        write_progress(write_progress_cwd.wd, stage, percent, eta)


def main():
    ap = argparse.ArgumentParser(description="M96 驱动工况循环 (dq+FOC, 对标 Lab/JMAG-RT/PLECS)")
    ap.add_argument("--workdir", default=os.path.join(HERE, "m96_runs"),
                    help="工作目录隔离 (默认 m96_runs)")
    ap.add_argument("--dt", type=float, default=None, help="步长 (白名单内)")
    ap.add_argument("--cycle", default=None, help="只跑指定名循环")
    ap.add_argument("--integrator", choices=["rk4", "euler"], default="rk4")
    ap.add_argument("--list", action="store_true", help="列出循环表退出")
    args = ap.parse_args()

    sp = load_drv()
    dt = args.dt if args.dt is not None else sp["dt_default"]
    if dt not in sp["dt_whitelist"]:
        raise ValueError(f"dt={dt} 不在白名单 {sp['dt_whitelist']} (禁任意步长)")
    if args.list:
        for c in sp["cycles"]:
            print(f"{c['name']}: kind={c['kind']}")
        for s in STANDARD_CYCLES:
            print(f"{s}: kind=standard")
        return 0

    os.makedirs(args.workdir, exist_ok=True)
    write_progress_cwd.wd = args.workdir
    write_progress(args.workdir, "start", 0.0)
    t0 = time.time()
    if args.cycle in STANDARD_CYCLES:
        # M98 标准循环: 准静态逐秒评估 (不经 FOC 时域链)
        r = run_standard_cycle(args.cycle, sp)
        p_loss_avg = (r["e_in_mot_MJ"] - r["e_out_MJ"]) * 1e6 / r["n_points"]
        r["thermal_proxy"] = thermal_proxy(p_loss_avg, sp)
        r["eta_ss_pct"] = r["eta_pct"]
        rows = [r]
        write_progress(args.workdir, f"standard:{args.cycle}", 95.0,
                       time.time() - t0)
        anchor_ok = r["infeasible_s"] == 0
        cycles = []
    else:
        cycles = [c for c in sp["cycles"]
                  if args.cycle is None or c["name"] == args.cycle]
        if not cycles:
            raise ValueError(f"未找到循环: {args.cycle} "
                             f"(合法: {[c['name'] for c in sp['cycles']]} "
                             f"+ {list(STANDARD_CYCLES)})")

        rows = []
        for i, c in enumerate(cycles):
            kind = c["kind"]
            if kind == "constant":
                r = run_constant_cycle(c, dt, sp, args.integrator)
            elif kind == "step":
                r = run_step_cycle(c, dt, sp, args.integrator)
            elif kind == "scan":
                r = run_scan_cycle(c, dt, sp, args.integrator)
            else:
                raise ValueError(f"未知循环类型: {kind}")
            r.update({"name": c["name"], "kind": kind,
                      "speed_rpm": c.get("speed_rpm"), "torque_Nm": c.get("torque_Nm"),
                      "t_end_s": c["t_end_s"],
                      "thermal_proxy": thermal_proxy(r["p_loss_avg_W"], sp)})
            r["status"] = "FAIL" if r["diverged"] else "PASS"
            rows.append(r)
            write_progress(args.workdir, f"cycle:{c['name']}",
                           (i + 1) / len(cycles) * 90.0 + 5.0, time.time() - t0)

    rated = next((r for r in rows if r["name"] == "rated_constant"), rows[0])
    is_std = bool(cycles) is False
    if not is_std:
        checks = ([p["iq_steady_check"] for r in rows for p in r.get("scan_points", [])
                   if p.get("iq_steady_check")]
                  + [r["iq_steady_check"] for r in rows if "iq_steady_check" in r])
        anchor_ok = bool(checks) and all(abs(c["dev_pct"]) < 1.0 for c in checks)
    any_fail = any(r["status"] == "FAIL" for r in rows)
    report = {
        "module": "M98" if is_std else "M96",
        "integrator": args.integrator, "dt_s": dt,
        "dt_whitelist_s": sp["dt_whitelist"],
        "constants_src": ["constants/drive_design.json"] +
                         ([f"constants/cycles/{args.cycle}.json",
                           "constants/map_design.json"] if is_std else []),
        "cycles": rows,
        "m92_consistency": None if is_std else m92_consistency(
            {"torque_Nm": rated.get("torque_Nm", 0.0) or 0.0}, sp,
            eta_ss_pct=rated.get("eta_ss_pct")),
        "anchor_gate_pct": 1.0,
        "anchor_check_pass": anchor_ok,
        "overall_status": "FAIL" if (any_fail or not anchor_ok) else "PASS",
        "caveats": (rows[0]["caveats"] if is_std else sp["caveats"] + [
            "step 循环分段间状态不续接 (每段独立从零速起仿, 口径披露)",
            "母线电流为平均模型 P/Udc, 无开关纹波",
        ]),
        "honest_disclosure": True, "elapsed_s": round(time.time() - t0, 2),
    }
    out = os.path.join(args.workdir, "drivecycle_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_progress(args.workdir, "done", 100.0, time.time() - t0)
    print(f"{'M98' if is_std else 'M96'} 驱动工况循环: {len(rows)} 循环, "
          f"overall={report['overall_status']}")
    for r in rows:
        if r["kind"] == "standard":
            print(f"  {r['name']} (standard): eta={r['eta_pct']:.2f}% "
                  f"dist={r['dist_km']}km vmax={r['v_max_kmh']}km/h "
                  f"infeasible={r['infeasible_s']}s regen_avail={r['e_regen_available_MJ']}MJ "
                  f"ΔT_proxy={r['thermal_proxy']['dT_proxy_K']}K status={r['status']}")
        else:
            print(f"  {r['name']} ({r['kind']}): eta={r['eta_pct']:.2f}% "
                  f"eta_ss={r.get('eta_ss_pct', 'n/a')}% "
                  f"Ibus_max={r['bus_i_max_A']:.2f}A Vbus={r['bus_v_max_V']:.0f}V "
                  f"iQ_dev={r.get('iq_steady_check', {}).get('dev_pct', 'n/a')}% "
                  f"ΔT_proxy={r['thermal_proxy']['dT_proxy_K']}K status={r['status']}")
    print(f"报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

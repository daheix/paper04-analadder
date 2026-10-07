#!/usr/bin/env python3
"""M98 效率 MAP 引擎驱动器 — id-iq 网格扫描 (对标 Motor-CAD Lab 效率 MAP / JMAG)

用法:
    python3 run_efficiency_map.py                          # 默认白名单网格/转速点
    python3 run_efficiency_map.py --workdir /tmp/m98a      # workdir 隔离

口径: map_spec.py (MTPA dTe/dβ=0 解析/电压极限椭圆/M92 损耗/marching squares);
      电磁链复用 M96 drv_spec (Te/Rs/Ld/Lq/λm/p); 网格与转速点 constants/map_design.json。
输出: <workdir>/efficiency_map.json + _progress.json (进度直通)。串行扫描 (无 Pool)。
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from drv_spec import em_torque, load_drv, p_cu_dq
from load_constants import map_design, materials, operating
from map_spec import (eta_of, iso_lines, load_map, mtpa_currents, mtpa_beta,
                      voltage_mag, voltage_ok)


def write_progress(workdir, stage, percent, eta_s=None):
    """_progress.json 进度直通 (C++ 1s 轮询口径)。"""
    with open(os.path.join(workdir, "_progress.json"), "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "percent": round(percent, 1),
                   "eta_s": eta_s, "ts": time.time()}, f, ensure_ascii=False)


def grid_axes(mp):
    """id/iq 坐标轴 (均匀 n 点, 端点闭区间)。"""
    n = int(mp["map_grid_n"])
    ids = [mp["map_id_min_A"] + (mp["map_id_max_A"] - mp["map_id_min_A"]) * i / (n - 1)
           for i in range(n)]
    iqs = [mp["map_iq_min_A"] + (mp["map_iq_max_A"] - mp["map_iq_min_A"]) * i / (n - 1)
           for i in range(n)]
    return ids, iqs


def scan_speed(speed_rpm, mp, dsp, loss):
    """单转速 id-iq 网格扫描 → eta/te 矩阵 + 弱磁边界点 + 峰值。"""
    we = dsp["p"] * speed_rpm * 2.0 * math.pi / 60.0
    ids, iqs = grid_axes(mp)
    eta_m, te_m, fw_pts = [], [], []
    peak = {"eta_pct": -1.0}
    umax_band = mp["map_Udc_V"] / math.sqrt(3.0) * mp["map_fw_band_frac"]
    for id_v in ids:
        row_e, row_t = [], []
        for iq in iqs:
            te = em_torque(id_v, iq, dsp["p"], dsp["lam"], dsp["Ld"], dsp["Lq"])
            feas = voltage_ok(id_v, iq, we, dsp["Ld"], dsp["Lq"], dsp["lam"],
                              mp["map_Udc_V"])
            e = eta_of(te, speed_rpm * 2.0 * math.pi / 60.0, id_v, iq, dsp["Rs"],
                       loss["kh"], loss["ke"], abs(we) / (2.0 * math.pi),
                       loss["bg1"], loss["v_fe"]) if feas else 0.0
            row_e.append(round(e, 4))
            row_t.append(round(te, 5))
            if feas:
                u = voltage_mag(id_v, iq, we, dsp["Ld"], dsp["Lq"], dsp["lam"])
                if umax_band <= u <= mp["map_Udc_V"] / math.sqrt(3.0):
                    fw_pts.append([round(id_v, 5), round(iq, 5)])
                if e > peak["eta_pct"]:
                    peak = {"eta_pct": round(e, 4), "id_A": round(id_v, 5),
                            "iq_A": round(iq, 5), "speed_rpm": speed_rpm,
                            "torque_Nm": round(te, 5)}
        eta_m.append(row_e)
        te_m.append(row_t)
    return {"speed_rpm": speed_rpm, "we_rad_s": round(we, 6),
            "id_A": [round(v, 5) for v in ids],
            "iq_A": [round(v, 5) for v in iqs],
            "eta_pct": eta_m, "torque_Nm": te_m,
            "fw_boundary_points": fw_pts,
            "peak": peak if peak["eta_pct"] > 0 else None,
            "max_torque_Nm": max(max(r) for r in te_m)}


def mtpa_trajectory(mp, dsp, loss):
    """MTPA 轨迹: Is 均匀 0→Imax, 参考转速下电压可行性标注。"""
    n = int(mp["map_grid_n"])
    imax = mp["map_imax_A"]
    w_ref = mp["map_cycle_speed_ref_rpm"]
    we = dsp["p"] * w_ref * 2.0 * math.pi / 60.0
    pts = []
    for k in range(n):
        is_amp = imax * k / (n - 1)
        if is_amp <= 0.0:
            pts.append({"Is_A": 0.0, "id_A": 0.0, "iq_A": 0.0, "beta_rad": 0.0,
                        "torque_Nm": 0.0,
                        "voltage_feasible": voltage_ok(0.0, 0.0, we, dsp["Ld"],
                                                       dsp["Lq"], dsp["lam"],
                                                       mp["map_Udc_V"])})
            continue
        id_v, iq = mtpa_currents(is_amp, dsp["Ld"], dsp["Lq"], dsp["lam"])
        te = em_torque(id_v, iq, dsp["p"], dsp["lam"], dsp["Ld"], dsp["Lq"])
        feas = voltage_ok(id_v, iq, we, dsp["Ld"], dsp["Lq"], dsp["lam"],
                          mp["map_Udc_V"])
        pts.append({"Is_A": round(is_amp, 5), "id_A": round(id_v, 5),
                    "iq_A": round(iq, 5),
                    "beta_rad": round(mtpa_beta(is_amp, dsp["Ld"], dsp["Lq"],
                                                dsp["lam"]), 9),
                    "torque_Nm": round(te, 5), "voltage_feasible": feas})
    return {"speed_ref_rpm": w_ref, "points": pts}


def iso_lines_speed(scan, mp):
    """单转速等效率线: 索引坐标 → (id, iq) 物理坐标, 附效率水平。"""
    ids, iqs = scan["id_A"], scan["iq_A"]
    out = []
    for lv in mp["map_iso_levels_pct"]:
        segs = []
        for (x1, y1), (x2, y2) in iso_lines(scan["eta_pct"], lv):
            segs.append([[round(ids[int(x1)] + (x1 % 1.0) * (ids[int(min(x1 + 1, len(ids) - 1))] - ids[int(x1)]), 5) if x1 < len(ids) - 1 else ids[-1],
                          round(iqs[int(y1)] + (y1 % 1.0) * (iqs[int(min(y1 + 1, len(iqs) - 1))] - iqs[int(y1)]), 5) if y1 < len(iqs) - 1 else iqs[-1]],
                         [round(ids[int(x2)] + (x2 % 1.0) * (ids[int(min(x2 + 1, len(ids) - 1))] - ids[int(x2)]), 5) if x2 < len(ids) - 1 else ids[-1],
                          round(iqs[int(y2)] + (y2 % 1.0) * (iqs[int(min(y2 + 1, len(iqs) - 1))] - iqs[int(y2)]), 5) if y2 < len(iqs) - 1 else iqs[-1]]])
        if segs:
            out.append({"level_pct": lv, "segments": segs})
    return out


def main():
    ap = argparse.ArgumentParser(
        description="M98 效率 MAP 引擎 (id-iq 网格扫描, 对标 Motor-CAD Lab / JMAG)")
    ap.add_argument("--workdir", default=os.path.join(HERE, "m98_runs"),
                    help="工作目录隔离 (默认 m98_runs)")
    ap.add_argument("--list", action="store_true", help="列出转速点退出")
    args = ap.parse_args()

    mp = load_map()
    dsp = load_drv()
    steel = materials()["steel_50JN350"]
    loss = {"kh": steel["KH"], "ke": steel["KE"], "bg1": mp["map_bg1_ref_T"],
            "v_fe": operating()["V_fe_m3"]}
    if args.list:
        for w in mp["map_speed_rpm_points"]:
            print(f"{w} rpm")
        return 0

    os.makedirs(args.workdir, exist_ok=True)
    write_progress(args.workdir, "start", 0.0)
    t0 = time.time()
    speeds = mp["map_speed_rpm_points"]
    scans = []
    for i, w in enumerate(speeds):
        sc = scan_speed(w, mp, dsp, loss)
        sc["iso_lines"] = iso_lines_speed(sc, mp)
        scans.append(sc)
        write_progress(args.workdir, f"scan:{w}rpm", (i + 1) / len(speeds) * 90.0,
                       time.time() - t0)

    peaks = [s["peak"] for s in scans if s["peak"]]
    peak = max(peaks, key=lambda p: p["eta_pct"]) if peaks else None
    mtpa = mtpa_trajectory(mp, dsp, loss)
    # 方向性检查: 同电流 Is=Imax 下, MTPA (含磁阻转矩) 转矩必须 ≥ id=0 纯 iq 转矩
    end = mtpa["points"][-1]
    te_id0 = em_torque(0.0, mp["map_imax_A"], dsp["p"], dsp["lam"],
                       dsp["Ld"], dsp["Lq"])
    mtpa_anchor_ok = end["torque_Nm"] >= te_id0 - 1e-9 and (
        mp["map_imax_A"] * math.hypot(end["id_A"] / mp["map_imax_A"],
                                      end["iq_A"] / mp["map_imax_A"])
        - mp["map_imax_A"]) < 1e-3
    fw_present = any(s["fw_boundary_points"] for s in scans
                     if s["speed_rpm"] > mp["map_cycle_speed_ref_rpm"])
    report = {
        "module": "M98", "constants_src": ["constants/map_design.json",
                                           "constants/drive_design.json",
                                           "constants/materials.json",
                                           "constants/operating_conditions.json"],
        "drive_params": {"Rs_ohm": dsp["Rs"], "Ld_H": dsp["Ld"], "Lq_H": dsp["Lq"],
                         "psi_m_Wb": dsp["lam"], "p": dsp["p"]},
        "loss_model": {"kh": loss["kh"], "ke": loss["ke"],
                       "bg1_ref_T": loss["bg1"], "v_fe_m3": loss["v_fe"],
                       "umax_V": round(mp["map_Udc_V"] / math.sqrt(3.0), 6)},
        "grid": {"n": mp["map_grid_n"], "id_range_A": [mp["map_id_min_A"], mp["map_id_max_A"]],
                 "iq_range_A": [mp["map_iq_min_A"], mp["map_iq_max_A"]]},
        "mtpa_trajectory": mtpa, "scans": scans, "peak_efficiency": peak,
        "iso_levels_pct": mp["map_iso_levels_pct"],
        "checks": {"mtpa_endpoint_matches_grid_max_Te": mtpa_anchor_ok,
                   "fw_boundary_present_above_base_speed": fw_present},
        "overall_status": "PASS" if (mtpa_anchor_ok and fw_present and peak) else "FAIL",
        "elapsed_s": round(time.time() - t0, 3),
        "caveats": mp["map_caveats"],
    }
    out = os.path.join(args.workdir, "efficiency_map.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    write_progress(args.workdir, "done", 100.0, time.time() - t0)
    print(f"M98 效率 MAP: {len(scans)} 转速点 × {mp['map_grid_n']}² 网格, "
          f"η峰={peak['eta_pct'] if peak else 'N/A'}% "
          f"@ {peak['speed_rpm'] if peak else '-'}rpm, "
          f"状态={report['overall_status']}, → {out}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

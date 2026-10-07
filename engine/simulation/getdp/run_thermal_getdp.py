#!/usr/bin/env python3
"""热路 LPTN — 基于 GetDP 真实磁密的热源 + 2D FEM 温度场 (v3, 修 1038°C 伪影)

伪影根因 (旧 coupling_solver.py):
  1. 铁耗喂了老 numpy 电磁链的错误磁密 (Bmax=6.4T, 真实 ~1.2T) → 铁耗 896 W;
  2. 同步电机转子铁耗应为 ~0, 旧代码把铁耗对半撒给定子+转子;
  3. 气隙导热取静止空气 0.026 W/mK, 转子热源穿不过去 → 磁体 1036°C。

v3 口径:
  - 铁耗: Steinmetz (kh·f + ke·f²)·|B|², B 取 GetDP b_map_dl_90.pos 逐单元幅值,
    仅定子铁心 (phys=7); 转子铁耗=0 (同步, 基波相对静止); 磁体涡流不计 (注记)。
  - 铜耗: R_ph = ρ·L_turn·N_ph/A_wire, I_rms = i_pk/√2, 三相。
  - 全部热源按 (真实 W) / L_stack → W/m³ (单位叠厚口径, 与 2D FEM 自洽)。
  - 对流边界按真实边长 + 端盖折算 h_end=1.25; 气隙有效导热 0.26 (thermal_solver)。
"""
import math
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from run_emag_getdp import GetDPMotorDriver, parse_bmap, js_to_i, P
from fem_check import read_msh2
from thermal_solver import ThermalFemSolver

# Steinmetz 系数 (50JN350 量级硅钢, 标定 50Hz@1T≈1.0 W/kg=7650 W/m³)
from load_constants import (materials as _mat_fn, operating as _op_fn,
                            physics as _ph_fn, cooling as _cooling_fn)
_mat, _op, _ph = _mat_fn(), _op_fn(), _ph_fn()
_COOL = _cooling_fn()
KH, KE = _mat["steel_50JN350"]["KH"], _mat["steel_50JN350"]["KE"]
RHO_CU = _mat["copper"]["RHO_20"]
FILL = _mat["copper"]["FILL"]
H_END = _op["h_end"]
SPEED_RPM = _op["speed_rpm"]
ALPHA_CU = _ph["ALPHA_CU"]
TAG_STATOR = 7
TAG_SLOTS = (6, 8, 9, 10, 11, 12)


def thermal_lptn(h_conv=15.0, bmap="b_map_dl_90.pos", verbose=True,
                 t_cu=None, js_peak=2.0e6):
    """LPTN 口径热路: 返回 dict(热源/温度场/各区温度)。h_conv=15 自然冷+辐射等效。
    t_cu: 铜温反馈 (°C) → R_ph(T)=R20·(1+α_Cu·(T−20)), 电磁-热耦合迭代用。
    js_peak: 槽电流密度峰值 A/m² (效率 Map 扫掠用), 默认额定 2e6。"""
    drv = GetDPMotorDriver()
    nd, tris = read_msh2(os.path.join(drv.wd, "motor_sector.msh"))
    # msh2 读出: nodes={nid:(x,y)} (nid 1-based), tris=[(n1,n2,n3,phys)]
    ids = sorted(nd)
    idx = {nid: i for i, nid in enumerate(ids)}
    nodes = np.array([nd[i] for i in ids])
    tris = [(idx[a], idx[b], idx[c], ph) for a, b, c, ph in tris]
    cen, area, _ = drv._elem_table()
    b = parse_bmap(os.path.join(drv.wd, bmap))
    bmap_d = {(round(x, 6), round(y, 6)): (bx, by) for x, y, bx, by in b}
    key = [(round(c[0], 6), round(c[1], 6)) for c in cen]
    B = np.array([bmap_d[k] for k in key])
    bmag = np.hypot(B[:, 0], B[:, 1])

    f = P["p"] * SPEED_RPM / 60.0          # 电频率 100 Hz @3000rpm
    # ---- 铁耗 (仅定子; 转子同步 ~0) ----
    fe_mask = np.array([t[3] == TAG_STATOR for t in tris])
    A_st = float(area[fe_mask].sum())
    # 局部过饱和单元 (槽口/磁极角 P0 可到 2-3T) 限幅到材料饱和 1.9T:
    # Steinmetz 线性叠加模型在深饱和区不再 B² 外推
    bmag_cl = np.clip(bmag, 0.0, 1.9)
    p_fe = (KH * f + KE * f * f) * bmag_cl ** 2       # W/m³
    P_fe = float((p_fe * area * fe_mask).sum() * P["L_stack"])

    # ---- 铜耗 ----
    slot_mask = np.array([t[3] in TAG_SLOTS for t in tris])
    A_slot = float(area[slot_mask].sum() / P["n_slots"])   # 单槽铜区面积 (网格)
    n_cond = int(2 * P["Nc"])                              # 双层: 2 线圈边 × Nc
    A_wire = A_slot * FILL / n_cond
    L_end = math.pi * 0.02                                 # 端部半周长估计 (跨 1 齿)
    L_turn = 2.0 * (P["L_stack"] + L_end)
    N_ph = 4 * P["Nc"]                                     # 每相 4 线圈串联
    R_ph20 = RHO_CU * L_turn * N_ph / A_wire
    # 铜电阻温度系数 α_Cu = 3.93e-3/K (20°C 基准, IEC 60028)
    R_ph = R_ph20 * (1.0 + ALPHA_CU * ((t_cu if t_cu is not None else 20.0) - 20.0))
    i_pk = js_to_i(js_peak)
    i_rms = i_pk / math.sqrt(2)
    P_cu = 3.0 * i_rms ** 2 * R_ph

    # ---- 热源 → thermal FEM ----
    m2t = {name: i + 1 for i, name in enumerate(
        ["air", "iron", "magnet_n", "magnet_s", "gap", "SlotAp", "SlotCn",
         "SlotBp", "SlotAn", "SlotCp", "SlotBn", "Outer"])}
    # 定子铁心 → stator_iron; 槽 → copper; 磁体 → permanent_magnet;
    # 转子铁/轴 → rotor_iron (导热路径, 无内热源)
    tri_mat = []
    mat = {}
    for t in tris:
        ph = t[3]
        if ph == TAG_STATOR:
            m = "stator_iron"
        elif ph in TAG_SLOTS:
            m = "copper_winding"
        elif ph in (3, 4):                 # magnet N/S
            m = "permanent_magnet"
        else:                              # 2=rotor iron, 1=shaft, 5=gap, 13=outer
            m = "rotor_iron"
        tri_mat.append(m)
        mat.setdefault(m, []).append(len(tri_mat) - 1)
    q_fe = P_fe / A_st / P["L_stack"] if A_st else 0.0
    q_cu = P_cu / (A_slot * P["n_slots"]) / P["L_stack"]
    hs = np.array([q_fe if m == "stator_iron"
                   else (q_cu if m == "copper_winding" else 0.0)
                   for m in tri_mat])
    # thermal_solver 节点/单元编号与 msh2 一致 (同源网格)
    th = ThermalFemSolver(nodes, [list(t[:3]) for t in tris], mat)
    th._precompute()
    res = th.solve(heat_sources=hs, T_ambient=40.0, h_conv=h_conv,
                   h_end_factor=H_END)

    out = {
        "speed_rpm": SPEED_RPM, "f_elec_Hz": f,
        "P_cu_W": round(P_cu, 3), "P_fe_W": round(P_fe, 3),
        "P_total_W": round(P_cu + P_fe, 3),
        "R_ph_ohm": round(R_ph, 3), "R_ph20_ohm": round(R_ph20, 3),
        "A_wire_mm2": round(A_wire * 1e6, 4),
        "i_rms_A": round(i_rms, 4), "L_turn_m": round(L_turn, 4),
        "B_max_stator_T": round(float(bmag[fe_mask].max()) if fe_mask.any() else 0, 4),
        "h_conv": h_conv, "h_end_factor": H_END,
        "T_max": round(float(res["T_max"]), 2),
        "T_avg": round(float(res["T_avg"]), 2),
        "T_winding": round(float(res["T_winding"]), 2),
        "T_magnet": round(float(res["T_magnet"]), 2),
        "T_stator": round(float(res["T_stator"]), 2),
        "T_rotor": round(float(res["T_rotor"]), 2),
        "hotspot_mm": [round(float(res["hotspot"][0]) * 1e3, 1),
                       round(float(res["hotspot"][1]) * 1e3, 1)],
    }
    if verbose:
        print(f"\n== 热路 LPTN ({SPEED_RPM:.0f} rpm, f={f:.0f} Hz, "
              f"h_conv={h_conv} W/m²K 自然冷+辐射, 端盖折算 {H_END}) ==")
        print(f"  损耗: 铜耗 {P_cu:.2f} W (R_ph={R_ph:.2f}Ω, I_rms={i_rms:.3f} A, "
              f"A_wire={A_wire*1e6:.3f} mm²) + 铁耗 {P_fe:.2f} W "
              f"(Steinmetz kh={KH} ke={KE}, Bmax_stator={out['B_max_stator_T']} T, 仅定子)")
        print(f"  温度: Tmax={out['T_max']}°C, 绕组={out['T_winding']}°C, "
              f"磁体={out['T_magnet']}°C, 定子={out['T_stator']}°C")
        print(f"  热点: ({out['hotspot_mm'][0]}, {out['hotspot_mm'][1]}) mm")
    return out


def main():
    """M92: 冷却拓扑枚举驱动 (constants/cooling.json) — natural/forced_air
    保持既有回归值 15/50 不变; 水冷套/油冷/蒸发冷却为 LPTN 等效对流口径
    (通道内对流+套壁导热折算定子外缘 h, 非 CFD, 逐拓扑披露 note)。"""
    outs = {}
    order = _COOL["cooling_report_order"]
    for i, key in enumerate(order):
        topo = _COOL["cooling_topologies"][key]
        h = topo["h_conv"]
        note = topo.get("note", "")
        if note:
            print(f"[cooling] {key} ({topo['label']}): h_eq={h} — {note}")
        outs[key] = thermal_lptn(h_conv=h)
    with open(os.path.join(HERE, "thermal_report.json"), "w", encoding="utf-8") as f:
        json.dump(outs, f, ensure_ascii=False, indent=2)
    print(f"\n→ thermal_report.json ({len(outs)} 拓扑: {', '.join(outs)})")
    return outs


if __name__ == "__main__":
    main()

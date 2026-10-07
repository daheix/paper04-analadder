#!/usr/bin/env python3
"""M98 效率 MAP 引擎口径卡 (doctest 可执行) — MTPA 解析 + 弱磁椭圆 + 损耗/等效率线

链路: map_spec.py (本卡) + run_efficiency_map.py (驱动器)
      + constants/map_design.json (map_* 白名单)
      → id-iq 网格扫描 → efficiency_map.json (网格+MTPA 轨迹+弱磁边界+等效率线+峰值效率点)

口径 (电磁链 — 复用 M96 drv_spec):
    Te = (3/2)·p·[λm·iq + (Ld − Lq)·id·iq];  ωe = p·ω

口径 (MTPA 轨迹 — dTe/dβ=0 解析解):
    电流角约定 id = −Is·sinβ, iq = Is·cosβ;
    Te(β) = (3/2)·p·[λm·Is·cosβ − (1/2)·(Lq−Ld)·Is²·sin2β]
    dTe/dβ = −(3/2)·p·[λm·Is·sinβ + (Lq−Ld)·Is²·cos2β] = 0
    → 2·(Lq−Ld)·Is·s² − λm·s − (Lq−Ld)·Is = 0, s = sinβ
    → s = [sqrt(λm² + 8·(Lq−Ld)²·Is²) − λm] / (4·(Lq−Ld)·Is)   (Lq>Ld 取正根)
    Ld = Lq (SPM) 时 β ≡ 0 (id=0)。

口径 (弱磁区 — 电压极限椭圆):
    ω·sqrt((Ld·id + λm)² + (Lq·iq)²) ≤ Umax, Umax = Udc/√3 (SVPWM 线性区, 与 M96 一致)

口径 (损耗 — 复用 M92):
    铜损 P_cu = (3/2)·Rs·(id² + iq²) ≡ 3·I_rms²·R_ph (幅值约定恒等, 与 M96 p_cu_dq 一致)
    铁损 P_fe = (KH·f + KE·f²)·Bg1²·V_fe, f = ωe/2π (M92 Steinmetz 口径, Bg1 取基线常数代理)
    效率 η = P_out/(P_out + P_cu + P_fe), P_out = Te·ω (电机本体口径, 未含逆变器/风摩)

口径 (等效率线): id-iq 平面 η 场 marching squares 线性插值提取 η=level 等值线段。

>>> spec = load_map()
>>> spec["map_Udc_V"], spec["map_grid_n"]
(300.0, 41)
>>> # ---- MTPA 锚点 (手工算例, 逐位一致): λm=0.1, Ld=0.002, Lq=0.003, Is=10 ----
>>> # s = [sqrt(0.01+8e-4) − 0.1]/0.04 = (√0.0108 − 0.1)/0.04
>>> round(mtpa_sin_beta(10.0, 0.002, 0.003, 0.1), 12)
0.098076211353
>>> round(mtpa_beta(10.0, 0.002, 0.003, 0.1), 12)
0.098234127441
>>> id_m, iq_m = mtpa_currents(10.0, 0.002, 0.003, 0.1)
>>> round(id_m, 12), round(iq_m, 12)
(-0.980762113533, 9.95178906914)
>>> # dTe/dβ = 0 逐位验证 (中心差分恰为 0.0)
>>> p, lam, ld, lq = 2, 0.1, 0.002, 0.003
>>> teb = lambda b: em_torque(-10.0*math.sin(b), 10.0*math.cos(b), p, lam, ld, lq)
>>> h = 1e-7
>>> (teb(0.09823412744140042 + h) - teb(0.09823412744140042 - h)) / (2*h)
0.0
>>> # SPM (Ld=Lq): MTPA 退化为 id=0
>>> mtpa_beta(10.0, 0.002, 0.002, 0.1)
0.0
>>> # Is→0 极限: β→0 (小电流 MTPA 趋近 id=0)
>>> abs(mtpa_beta(1e-9, 0.002, 0.003, 0.1)) < 1e-6
True
>>> # ---- 电压极限椭圆 (手工算例): ωe=100, id=0, iq=10/3 ----
>>> # u = 100·sqrt(0.1² + (0.003·10/3)²) = 10.0498756211 ≤ Umax=300/√3 → feasible
>>> round(voltage_mag(0.0, 10.0/3.0, 100.0, 0.002, 0.003, 0.1), 10)
10.0498756211
>>> voltage_ok(0.0, 10.0/3.0, 100.0, 0.002, 0.003, 0.1, 300.0)
True
>>> voltage_ok(0.0, 60.0, 10000.0*2*math.pi/60.0*2, 0.002, 0.003, 0.1, 300.0)
False
>>> # ---- 损耗/效率锚点: Rs=0.5, f=50Hz, Bg1=0.9155, V_fe=1e-3, KH=150, KE=0.05 ----
>>> round(p_cu_dq(id_m, iq_m, 0.5), 9)
75.0
>>> round(p_fe_steinmetz(150.0, 0.05, 50.0, 0.9155, 1e-3), 12)
6.39081940625
>>> # η = Pout/(Pout+Pcu+Pfe), Pout=Te·ω, ω=ωe/p=50 rad/s
>>> te = em_torque(id_m, iq_m, p, lam, ld, lq)
>>> round(te, 9)
3.014817734
>>> # ω = ωe/p = 100/2 = 50π rad/s
>>> round(eta_of(te, 50.0*math.pi, id_m, iq_m, 0.5, 150.0, 0.05, 50.0, 0.9155, 1e-3), 9)
85.333858641
>>> # 零转矩点效率定义为 0 (无输出)
>>> eta_of(0.0, 50.0, 0.0, 0.0, 0.5, 150.0, 0.05, 50.0, 0.9155, 1e-3)
0.0
>>> # ---- 等效率线 (marching squares): 常数场 → 无等值线; 线性场 → 提取有效段 ----
>>> iso_lines([[50.0]*3]*3, 60.0)
[]
>>> segs = iso_lines([[0.0, 10.0], [0.0, 10.0]], 5.0)
>>> len(segs), round(segs[0][0][0], 6), round(segs[0][0][1], 6)
(1, 1.0, 0.5)
>>> # ---- 非法输入 fail-fast ----
>>> mtpa_beta(-1.0, 0.002, 0.003, 0.1)
Traceback (most recent call last):
    ...
ValueError: 电流幅值必须为正: -1.0
>>> voltage_ok(0.0, 1.0, 100.0, 0.002, 0.003, 0.1, -300.0)
Traceback (most recent call last):
    ...
ValueError: Udc 必须为正: -300.0
>>> load_map_bad()
Traceback (most recent call last):
    ...
KeyError: 'map_design.json 缺字段: map_grid_n'
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import map_design
from drv_spec import em_torque, p_cu_dq  # 复用 M96 dq 链口径 (唯一实现)

_MAP_REQ_KEYS = [
    "map_Udc_V", "map_imax_A", "map_id_min_A", "map_id_max_A",
    "map_iq_min_A", "map_iq_max_A", "map_grid_n", "map_speed_rpm_points",
    "map_iso_levels_pct", "map_fw_band_frac", "map_bg1_ref_T",
    "map_cycle_speed_ref_rpm", "map_caveats",
]


def load_map():
    """map_* 白名单 (fail-fast) → spec dict。"""
    raw = map_design()
    for k in _MAP_REQ_KEYS:
        if k not in raw:
            raise KeyError(f"map_design.json 缺字段: {k}")
    return {k: raw[k] for k in _MAP_REQ_KEYS} | {"caveats": raw["map_caveats"], "raw": raw}


def load_map_bad():
    """doctest 用: 缺字段白名单装载必须 fail-fast。"""
    raw = dict(map_design())
    del raw["map_grid_n"]
    for k in _MAP_REQ_KEYS:
        if k not in raw:
            raise KeyError(f"map_design.json 缺字段: {k}")
    return raw


def mtpa_sin_beta(is_amp, ld, lq, lam):
    """MTPA 电流角正弦 s=sinβ = [sqrt(λm²+8·ΔL²·Is²) − λm]/(4·ΔL·Is), ΔL=Lq−Ld>0。"""
    if is_amp <= 0:
        raise ValueError(f"电流幅值必须为正: {is_amp}")
    dl = lq - ld
    if dl <= 0.0:
        return 0.0
    return (math.sqrt(lam * lam + 8.0 * dl * dl * is_amp * is_amp) - lam) \
        / (4.0 * dl * is_amp)


def mtpa_beta(is_amp, ld, lq, lam):
    """MTPA 电流角 β = asin(s); SPM (Ld≥Lq) 退化为 0。"""
    return math.asin(max(-1.0, min(1.0, mtpa_sin_beta(is_amp, ld, lq, lam))))


def mtpa_currents(is_amp, ld, lq, lam):
    """MTPA 电流 (id, iq) = (−Is·sinβ, Is·cosβ)。"""
    b = mtpa_beta(is_amp, ld, lq, lam)
    return -is_amp * math.sin(b), is_amp * math.cos(b)


def voltage_mag(id_v, iq, we, ld, lq, lam):
    """电压极限椭圆半径: ω·sqrt((Ld·id+λm)² + (Lq·iq)²)。"""
    return abs(we) * math.hypot(ld * id_v + lam, lq * iq)


def voltage_ok(id_v, iq, we, ld, lq, lam, udc):
    """弱磁判据: ω·sqrt((Ld·id+λm)²+(Lq·iq)²) ≤ Udc/√3 (SVPWM 线性区)。"""
    if udc <= 0:
        raise ValueError(f"Udc 必须为正: {udc}")
    return voltage_mag(id_v, iq, we, ld, lq, lam) <= udc / math.sqrt(3.0)


def p_fe_steinmetz(kh, ke, f_hz, bg1, v_fe):
    """M92 Steinmetz 铁损口径: (KH·f + KE·f²)·Bg1²·V_fe [W]。"""
    return (kh * f_hz + ke * f_hz * f_hz) * bg1 * bg1 * v_fe


def eta_of(te, w_mech, id_v, iq, rs, kh, ke, f_hz, bg1, v_fe):
    """效率 η (%) = Pout/(Pout+Pcu+Pfe)·100; Pout≤0 → 0。"""
    p_out = te * w_mech
    if p_out <= 0.0:
        return 0.0
    p_loss = p_cu_dq(id_v, iq, rs) + p_fe_steinmetz(kh, ke, f_hz, bg1, v_fe)
    return p_out / (p_out + p_loss) * 100.0


def iso_lines(field, level):
    """marching squares 等值线段提取。

    field[i][j] 为网格值 (i 行 id 序, j 列 iq 序), 返回 [[(x1,y1),(x2,y2)], ...]
    线性插值, x=行索引, y=列索引 (坐标换算由调用方完成)。
    """
    segs = []
    ni = len(field)
    nj = len(field[0]) if ni else 0

    def interp(p0, v0, p1, v1):
        t = (level - v0) / (v1 - v0)
        return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))

    for i in range(ni - 1):
        for j in range(nj - 1):
            corners = [(i, j, field[i][j]), (i + 1, j, field[i + 1][j]),
                       (i + 1, j + 1, field[i + 1][j + 1]),
                       (i, j + 1, field[i][j + 1])]
            pts = []
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                va, vb = corners[a][2], corners[b][2]
                if (va < level) != (vb < level):
                    pts.append(interp(corners[a][:2], va, corners[b][:2], vb))
            if len(pts) >= 2:
                segs.append([pts[0], pts[1]])
    return segs


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"map_spec doctest: {r.attempted} assertions, failed={r.failed}")
    raise SystemExit(1 if r.failed else 0)

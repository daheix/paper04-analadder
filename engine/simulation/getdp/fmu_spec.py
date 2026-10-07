#!/usr/bin/env python3
"""M97 ROM/FMU 导出口径卡 (doctest 可执行) — FMI 2.0 Co-Simulation + 降阶 dq 2 状态

链路: fmu_spec.py (本卡) + run_fmu_export.py (导出器)
      + constants/fmu_design.json (fmu_* 白名单) + constants/drive_design.json (drv_* 继承)
      → 自足 FMU: modelDescription.xml + sources/rom_core.c (C89) + resources/ 参考实现
        + zip 打包 → rom_report.json

FMI 2.0 Co-Simulation 规范要点 (本链自足实现口径, 无第三方 FMI SDK):
    - modelDescription.xml 根元素 fmiModelDescription, 必备属性 fmiVersion="2.0"/
      modelName/guid/generationTool/numberOfEventIndicators/numberOfContinuousStates;
    - <CoSimulation modelIdentifier=...> 声明 Co-Simulation 工具类型 (FMU 自带求解器,
      主控每通信步调 fmi2DoStep, 本链以 C89 数值核 rom_step 承担该角色);
    - <DefaultExperiment startTime/stopTime/stepSize> 给默认实验区间与通信步长;
    - <ModelVariables> 内 ScalarVariable 按 valueReference 编号,
      causality ∈ {input, parameter, output}, Real 变量带 start 初值;
    - 连续状态数 = 3: dq 机电 2 状态 (iq, ω) + 热合并节点 1 状态 (T_wound)。

降阶口径 (dq 2 状态 + LPTN 节点热容合并, 全披露):
    - id ≡ 0: M96 全阶 dq ODE 的 id 动态被 FOC d 轴电流环钳位 (id*=0),
      Ld/Rs=4 ms 电时间常数 ≪ 机电时间常数, 残余 id≈0 → 消去该状态;
    - dq 机电方程 (x=(iq, ω)):
          Lq·diq/dt = uq − Rs·iq − ωe·λm ,  ωe = p·ω,
          Te = (3/2)·p·λm·iq ,  J·dω/dt = Te − TL ;
    - 控制与 M96 一致: 转速环离散增量 PI → iq* (限幅 ±iq_max),
      q 电流环离散增量 PI + 前馈 uq = uq_PI + ωe·λm (d 轴 ud = −ωe·Lq·iq 纯前馈,
      不回灌状态); SVPWM 线性区 |u| ≤ Udc/√3 等比限幅; 控制量按步起点冻结后 RK4 植物;
    - 热 LPTN 合并: 绕组铜/定子铁心/转子-磁体三节点热容合并为单一集总节点
      (C_th = Σ m_i·c_i, 节点间接触导热 ≫ 表面对流 → 单节点等效, 披露):
          C_th·dT_wound/dt = P_cu + P_fe − (T_wound − T_amb)/R_th ,
      P_cu = 1.5·Rs·iq², P_fe = P_hyst·(f/f0) + P_eddy·(f/f0)², f = ωe/2π,
      R_th = 1/(h·S) (cooling.json natural h + 默认设计 S=π·D_so·L_stack);
    - 输出变量: λm/iq/ω/T_wound (+ω_rpm/Te/P_cu/P_fe 派生)。

>>> spec = load_fmu()
>>> spec["fmu_fmi_version"], spec["fmu_fmi_type"]
('2.0', 'CoSimulation')
>>> spec["fmu_model_identifier"]
'ChinaSimMotorROM'
>>> spec["fmu_dq_state_names"]
['iq_A', 'omega_rad_s']
>>> # ---- 热容合并规则: 单集总节点 C_th = Σ m_i·c_i (白名单直给) ----
>>> round(merged_c_rule([800.0, 1000.0, 200.0]), 9)
2000.0
>>> # R_th = 1/(h·S): h=15 自然冷, S=π·0.1·0.1=0.0314159... → 2.122066
>>> import math
>>> round(lumped_r_th(15.0, math.pi * 0.1 * 0.1), 6)
2.122066
>>> # ---- ROM 稳态锚点 (与 M96 同口径逐位一致) ----
>>> # id=0: iq = T/(1.5·p·λm); T=10Nm → 33.333333
>>> round(iq_from_torque(10.0, 2, 0.1), 6)
33.333333
>>> # ω = 3000rpm → 314.159265 rad/s
>>> round(rpm_to_rad_s(3000.0), 6)
314.159265
>>> # 稳态 uq = Rs·iq + ωe·λm = 50/3 + 10 (ωe=100, 与 drv_spec 锚点同口径)
>>> round(0.5 * 100.0/3.0 + 100.0 * 0.1, 6)
26.666667
>>> # ---- ROM 植物导数锚点: λm=0, iq=0, uq=0 → diq=0; dω/dt=(Te−TL)/J=−1200 ----
>>> sp0 = rom_spec_lite(lam=0.0)
>>> diq, dw, dtw = rom_derivs(0.0, 0.0, 25.0, 6.0, 0.0, sp0)
>>> round(diq, 9), round(dw, 9)
(0.0, -1200.0)
>>> # 热: dtw = (P_cu+P_fe − (T−Tamb)/R)/C, T=Tamb 时 = P/C
>>> sp1 = rom_spec_lite(pcu=900.0, pfe=100.0)
>>> round(rom_derivs(0.0, 0.0, 25.0, 0.0, 0.0, sp1)[2], 9)
0.5
>>> # ---- RK4 步进锚点 (与 drv_spec 同机制): λm=0 退化, 一步 ω = −0.12 ----
>>> sp2 = rom_spec_lite(lam=0.0)
>>> st = rom_rk4_step((0.0, 0.0, 25.0), 6.0, sp2, 1e-4)
>>> round(st[0], 9), round(st[1], 9), round(st[2], 9)
(0.0, -0.12, 25.0)
>>> # ---- 相对误差口径: 分母 = max(|ref|, 绝对下限) (全披露) ----
>>> round(rel_err(1.0001, 1.0, 0.01) * 100.0, 6)
0.01
>>> round(rel_err(1.1, 0.0, 0.01) * 100.0, 3)
11000.0
>>> # ---- FMU 变量表: input+parameter+output 编号连续, 输出含 λm/iq/ω/T_wound ----
>>> from drv_spec import load_drv
>>> fs = load_fmu(); dv = load_drv()["raw"]
>>> vs = fmu_variables(fs, dv)
>>> vs[0]
{'kind': 'input', 'name': 'TL_Nm', 'vr': 0, 'unit': 'Nm', 'start': 0.0}
>>> names = [v["name"] for v in vs]
>>> all(n in names for n in ["lam_m_Wb", "iq_A", "omega_rad_s", "T_wound_C"])
True
>>> kinds = [v["kind"] for v in vs]
>>> kinds.count("input"), kinds.count("parameter"), kinds.count("output")
(1, 16, 8)
>>> [v["vr"] for v in vs] == list(range(len(vs)))
True
>>> # ---- mode 模式枚举 (M99): 末位参数, 0=analytic_dq | 1=lut_rom ----
>>> vs[-9]
{'kind': 'parameter', 'name': 'mode', 'vr': 16, 'unit': '1', 'start': 0.0}
>>> vs[-9]["vr"] + 1 == vs[-8]["vr"]
True
>>> # ---- modelDescription.xml 关键字段断言 ----
>>> xml = build_model_description(fs, dv, thm_r=2.1220659078919377)
>>> 'fmiVersion="2.0"' in xml
True
>>> '<CoSimulation modelIdentifier="ChinaSimMotorROM"' in xml
True
>>> 'numberOfContinuousStates="3"' in xml
True
>>> 'numberOfEventIndicators="0"' in xml
True
>>> '<DefaultExperiment startTime="0' in xml and 'stepSize="0.0001"' in xml
True
>>> xml.count('<ScalarVariable') == len(fmu_variables(fs, dv))
True
>>> # 非法输入 fail-fast (禁静默兜底)
>>> merged_c_rule([])
Traceback (most recent call last):
    ...
ValueError: 热容列表为空
>>> lumped_r_th(0.0, 0.03)
Traceback (most recent call last):
    ...
ValueError: h·S 必须为正: h=0.0, S=0.03
>>> rpm_to_rad_s(-1.0)
Traceback (most recent call last):
    ...
ValueError: 转速必须非负: -1.0
"""
import json
import math
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import fmu_design
from drv_spec import pi_update, svpwm_limit, iq_from_torque, p_cu_dq, p_fe_proxy

_FMUREQ_KEYS = [
    "fmu_fmi_version", "fmu_fmi_type", "fmu_model_name", "fmu_model_identifier",
    "fmu_generation_tool", "fmu_comm_step_default_s", "fmu_default_stop_s",
    "fmu_dq_state_names", "fmu_thermal_state_names", "fmu_output_names",
    "fmu_thm_c_j_per_k", "fmu_thm_t_amb_c", "fmu_rel_err_tol_pct",
    "fmu_iq_abs_floor_a", "fmu_w_abs_floor_rad_s", "fmu_thm_t_abs_floor_c",
    "fmu_caveats",
]


def load_fmu():
    """fmu_* 白名单 (fail-fast) → spec dict (缺字段直接 KeyError, 禁静默兜底)。"""
    raw = fmu_design()
    for k in _FMUREQ_KEYS:
        if k not in raw:
            raise KeyError(f"fmu_design.json 缺字段: {k}")
    out = dict(raw)
    out["caveats"] = raw["fmu_caveats"]
    return out


def rpm_to_rad_s(rpm):
    """n [rpm] → ω [rad/s] (机械)。"""
    if rpm < 0:
        raise ValueError(f"转速必须非负: {rpm}")
    return rpm * 2.0 * math.pi / 60.0


def merged_c_rule(c_list):
    """LPTN 节点热容合并规则: 单集总节点 C_th = Σ m_i·c_i (直给各节点热容)。"""
    if not c_list:
        raise ValueError("热容列表为空")
    return float(sum(c_list))


def lumped_r_th(h_conv, s):
    """合并节点对流热阻 R_th = 1/(h·S) [K/W]。"""
    if h_conv <= 0 or s <= 0 or h_conv * s <= 0:
        raise ValueError(f"h·S 必须为正: h={h_conv}, S={s}")
    return 1.0 / (h_conv * s)


def rom_spec_lite(lam=0.1, p=2, lq=0.003, rs=0.5, j=0.005, udc=300.0,
                  kp_w=0.8, ki_w=20.0, kp_i=2.0, ki_i=500.0, iq_max=60.0,
                  pfe_hyst=15.0, pfe_eddy=25.0, pfe_fref=50.0,
                  thm_c=2000.0, thm_r=2.1220659078919377, thm_tamb=25.0,
                  pcu=0.0, pfe=0.0):
    """测试/锚点用 ROM spec (run_fmu_export 用 build_rom_spec 装配真参数)。"""
    return {"lam": lam, "p": int(p), "Lq": lq, "Rs": rs, "J": j, "Udc": udc,
            "kp_w": kp_w, "ki_w": ki_w, "kp_i": kp_i, "ki_i": ki_i,
            "iq_max": iq_max, "pfe_hyst": pfe_hyst, "pfe_eddy": pfe_eddy,
            "pfe_fref": pfe_fref, "thm_C": thm_c, "thm_R": thm_r,
            "thm_tamb": thm_tamb, "_pcu_override": pcu, "_pfe_override": pfe,
            "_integ_w": 0.0, "_integ_q": 0.0, "_w_ref": 0.0}


def build_rom_spec(drv, thm_c, thm_r, thm_tamb):
    """从 M96 drv spec (load_drv) + 热合并参数装配 ROM spec (口径唯一继承)。"""
    return {"lam": drv["lam"], "p": drv["p"], "Lq": drv["Lq"], "Rs": drv["Rs"],
            "J": drv["J"], "Udc": drv["Udc"], "kp_w": drv["kp_w"],
            "ki_w": drv["ki_w"], "kp_i": drv["kp_i"], "ki_i": drv["ki_i"],
            "iq_max": drv["iq_max"], "pfe_hyst": drv["pfe_hyst"],
            "pfe_eddy": drv["pfe_eddy"], "pfe_fref": drv["pfe_fref"],
            "thm_C": thm_c, "thm_R": thm_r, "thm_tamb": thm_tamb,
            "_pcu_override": 0.0, "_pfe_override": 0.0,
            "_integ_w": 0.0, "_integ_q": 0.0, "_w_ref": 0.0}


def rom_derivs(iq, w, t_w, tl, uq, sp):
    """降阶 ROM 右端 ODE (id≡0): (diq, dω, dT_wound)。

    Lq·diq/dt = uq − Rs·iq − ωe·λm;  dω/dt = (1.5·p·λm·iq − TL)/J;
    C·dT/dt = P_cu + P_fe − (T−Tamb)/R  (P 按步起点状态冻结)。
    """
    we = sp["p"] * w
    te = 1.5 * sp["p"] * sp["lam"] * iq
    dw = (te - tl) / sp["J"]
    diq = (uq - sp["Rs"] * iq - we * sp["lam"]) / sp["Lq"]
    if sp["_pcu_override"] > 0.0 or sp["_pfe_override"] > 0.0:
        p_cu, p_fe = sp["_pcu_override"], sp["_pfe_override"]
    else:
        p_cu = p_cu_dq(0.0, iq, sp["Rs"])
        f_hz = abs(we) / (2.0 * math.pi)
        p_fe = p_fe_proxy(sp["pfe_hyst"], sp["pfe_eddy"], f_hz, sp["pfe_fref"])
    dtw = (p_cu + p_fe - (t_w - sp["thm_tamb"]) / sp["thm_R"]) / sp["thm_C"]
    return diq, dw, dtw


def rom_control(w_ref, iq, w, integ_w, integ_q, sp, dt):
    """ROM 一步控制量 (步起点计算, RK4 子步冻结 — 与 M96 同口径):

    转速环 PI → iq* (限幅); q 电流环 PI + 前馈 → uq; SVPWM Udc/√3 限幅。
    返回 (iq_ref, uq, scale, integ_w_new, integ_q_new)。
    """
    vw, integ_w = pi_update(sp["kp_w"], sp["ki_w"], dt, w_ref - w, integ_w,
                            out_max=sp["iq_max"])
    iq_ref = max(-sp["iq_max"], min(sp["iq_max"], vw))
    uq_pi, integ_q = pi_update(sp["kp_i"], sp["ki_i"], dt, iq_ref - iq,
                               integ_q, out_max=sp["Udc"])
    we = sp["p"] * w
    uq_cmd = uq_pi + we * sp["lam"]
    _, uq, scale = svpwm_limit(0.0, uq_cmd, sp["Udc"])
    return iq_ref, uq, scale, integ_w, integ_q


def rom_rk4_step(state, tl, sp, dt):
    """ROM 一步 RK4: state=(iq, w, t_w); 控制量步起点冻结, PI 积分按步进一次。"""
    iq, w, t_w = state
    w_ref = sp["_w_ref"]
    _, uq, _, integ_w, integ_q = rom_control(w_ref, iq, w,
                                             sp["_integ_w"], sp["_integ_q"],
                                             sp, dt)
    k1 = rom_derivs(iq, w, t_w, tl, uq, sp)
    k2 = rom_derivs(iq + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1],
                    t_w + 0.5 * dt * k1[2], tl, uq, sp)
    k3 = rom_derivs(iq + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1],
                    t_w + 0.5 * dt * k2[2], tl, uq, sp)
    k4 = rom_derivs(iq + dt * k3[0], w + dt * k3[1], t_w + dt * k3[2],
                    tl, uq, sp)
    return (iq + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            w + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            t_w + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
            integ_w, integ_q)


def rel_err(a, ref, abs_floor):
    """相对误差口径: |a−ref| / max(|ref|, abs_floor) (下限防 0/0, 全披露)。"""
    if abs_floor <= 0:
        raise ValueError(f"绝对下限必须为正: {abs_floor}")
    return abs(a - ref) / max(abs(ref), abs_floor)


# ---- FMU 变量表 (valueReference 连续编号, input→parameter→output) ----

def fmu_variables(fs, drv_raw):
    """FMU 变量表: vr 0 = 负载转矩输入; vr 1..15 = drv_*/thm 参数;
    vr 16 = mode 模式枚举 (M99: 0=analytic_dq 解析 dq | 1=lut_rom 磁链 MAP LUT,
    并列可选, 接口/状态数不变, 仅切换 λ 来源); 输出 = λm/iq/ω/ω_rpm/Te/T_wound/P_cu/P_fe。"""
    pmap = [
        ("drv_Rs_ohm", "Rs", "Ohm"), ("drv_Lq_H", "Lq", "H"),
        ("drv_p", "p", "1"), ("drv_psi_m_Wb", "lam", "Wb"),
        ("drv_J_kg_m2", "J", "kg.m2"), ("drv_Udc_V", "Udc", "V"),
        ("drv_kp_w", "kp_w", "1"), ("drv_ki_w", "ki_w", "1"),
        ("drv_kp_i", "kp_i", "1"), ("drv_ki_i", "ki_i", "1"),
        ("drv_iq_max_A", "iq_max", "A"),
        ("drv_pfe_hyst_W_50hz", "pfe_hyst", "W"),
        ("drv_pfe_eddy_W_50hz", "pfe_eddy", "W"),
        ("drv_pfe_freq_ref_hz", "pfe_fref", "Hz"),
        ("thm_T_amb_C", "thm_tamb", "K"),
    ]
    short = {"Rs": "drv_Rs_ohm", "Lq": "drv_Lq_H", "p": "drv_p",
             "lam": "drv_psi_m_Wb", "J": "drv_J_kg_m2", "Udc": "drv_Udc_V",
             "kp_w": "drv_kp_w", "ki_w": "drv_ki_w", "kp_i": "drv_kp_i",
             "ki_i": "drv_ki_i", "iq_max": "drv_iq_max_A",
             "pfe_hyst": "drv_pfe_hyst_W_50hz", "pfe_eddy": "drv_pfe_eddy_W_50hz",
             "pfe_fref": "drv_pfe_freq_ref_hz"}
    vs = [{"kind": "input", "name": "TL_Nm", "vr": 0, "unit": "Nm", "start": 0.0}]
    vr = 1
    for name, key, unit in pmap:
        start = (fs["fmu_thm_t_amb_c"] if key == "thm_tamb"
                 else drv_raw[short[key]])
        vs.append({"kind": "parameter", "name": name, "vr": vr,
                   "unit": unit, "start": start})
        vr += 1
    mode_names = fs["fmu_mode_names"]
    if fs["fmu_default_mode"] not in mode_names:
        raise ValueError(f"缺字段/非法模式: {fs['fmu_default_mode']}")
    vs.append({"kind": "parameter", "name": "mode", "vr": vr, "unit": "1",
               "start": float(mode_names.index(fs["fmu_default_mode"]))})
    vr += 1
    for name in fs["fmu_output_names"]:
        vs.append({"kind": "output", "name": name, "vr": vr, "unit": "-", "start": 0.0})
        vr += 1
    return vs


def build_model_description(fs, drv_raw, thm_r, guid=None):
    """生成 FMI 2.0 Co-Simulation modelDescription.xml (自足, 无模板依赖)。

    thm_r: 合并节点热阻 (R_th=1/(h·S) 导出值, 显式传入禁静默缺省);
    guid: FMU 实例 GUID, 缺省稳定 UUID5 (同输入同 GUID, 可复现)。
    """
    g = guid or str(uuid.uuid5(uuid.NAMESPACE_URL,
                               fs["fmu_model_name"] + "|" + fs["fmu_generation_tool"]))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<fmiModelDescription fmiVersion="2.0"'
             f' modelName="{fs["fmu_model_name"]}"'
             f' guid="{g}"'
             f' generationTool="{fs["fmu_generation_tool"]}"'
             ' numberOfEventIndicators="0"'
             ' numberOfContinuousStates="3">',
             '  <CoSimulation'
             f' modelIdentifier="{fs["fmu_model_identifier"]}"'
             ' canHandleVariableCommunicationStepSize="true"'
             ' needsExecutionTool="false"'
             ' canInterpolateInputs="false"'
             ' maxOutputDerivativeOrder="0"/>',
             '  <DefaultExperiment'
             f' startTime="0" stopTime="{fs["fmu_default_stop_s"]:g}"'
             f' stepSize="{fs["fmu_comm_step_default_s"]:g}"/>',
             '  <ModelVariables>']
    vs = fmu_variables(fs, drv_raw)
    for v in vs:
        caus = {"input": "input", "parameter": "parameter",
                "output": "output"}[v["kind"]]
        lines.append(f'    <ScalarVariable name="{v["name"]}"'
                     f' valueReference="{v["vr"]}" causality="{caus}"'
                     + (' initial="exact"' if v["kind"] == "output" else "") + '>')
        lines.append(f'      <Real start="{v["start"]:g}"/>')
        lines.append('    </ScalarVariable>')
    lines.append('  </ModelVariables>')
    lines.append('  <ModelStructure>')
    for v in vs:
        if v["kind"] == "output":
            lines.append(f'    <Outputs><Unknown index="{vs.index(v) + 1}"/></Outputs>')
    lines.append('  </ModelStructure>')
    lines.append('</fmiModelDescription>')
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"fmu_spec doctest: {r.attempted} assertions, failed={r.failed}")
    raise SystemExit(1 if r.failed else 0)

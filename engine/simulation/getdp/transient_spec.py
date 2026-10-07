#!/usr/bin/env python3
"""M86 瞬态电磁链口径卡 (doctest 可执行) — 数值口径唯一事实源

链路: run_emag_transient.py → generate_geo.py(rotor_angle_deg 剪切带步进)
      → motor_mag_t.pro (GetDP TimeLoopTheta + Picard) → transient_report.json

口径:
- 时间推进: GetDP TimeLoopTheta[t0, t_end, dt, θ]，θ=1 隐式 Euler (θ∈(0.5,1] 合法;
  θ=0.5 Crank-Nicolson — 时间相关源在弱式中需手工分裂, v1 不启用)。
- 瞬态方程: ∮ ν(B)∇a·∇δa + σ·∂a/∂t·δa = ∫ hc·δa − ∫ js(t)·δa；
  σ 默认 0 (σ_magnet/σ_iron 由 operating_conditions.json transient_* 给出)。
- 时间源: js(t_k) 以 $Time 显式写入 Integral 项 — 官方文档口径: 仅隐式 Euler
  下时间相关源在当前步取值正确 (getdp.texi TimeLoopTheta Warning)。
- 非线性: 每时间步内层 Picard (NL_tol_rel / NL_iter_max, 与静磁同口径)。
- 转子旋转 (v1 剪切带步进口径): 全局极网格角向线不变; 转子环 (shaft 外圆/R_ri/
  R_rm) 节点坐标旋转 Δ=rotor_angle_deg, 磁体/转子材料按旋转后中角判定; 气隙带
  单元剪切为斜四边形 (保形网格, 无滑移面重剖分)。步长约束: Δ ≤ dtheta_max
  (网格最大角距), 由 check_rotation_step 强制。
- 对齐验收: Δ=0、i(t)=0 时瞬态网格/矩阵与静磁逐位一致 →
  |Bg1_transient − Bg1_static| / Bg1_static ≤ BGL_TOL (1%)。
- 进度埋点: _progress.json 键 schema 由 progress_keys() 定义 (与静磁链同)。

>>> spec = load_transient_spec()
>>> spec["theta"]                                  # θ=1 隐式 Euler
1.0
>>> round(electrical_freq_hz(3000.0, 4), 6)
100.0
>>> round(time_step_s(1.0/100.0, 40), 10) == round(1.0/100.0/40, 10)
True
>>> check_rotation_step(2.4, 2.5)                  # Δ ≤ dtheta_max 合法
True
>>> check_rotation_step(2.6, 2.5)
Traceback (most recent call last):
    ...
ValueError: 转子步进角 2.6° 超过网格最大角距 2.5° (剪切带单元翻转风险)
>>> progress_keys()
['stage', 'percent', 'elapsed_s', 'eta_s', 'msg', 'timestamp']
>>> round(pole_pitch_deg(4), 6)
90.0
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONST = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "constants")

# 验收判据 (口径卡级, 非业务数值): Bg1 对齐容差 = 用户验收线 1%
BGL_TOL = 0.01


def _operating():
    with open(os.path.join(_CONST, "operating_conditions.json"),
              encoding="utf-8") as f:
        return json.load(f)


def load_transient_spec():
    """瞬态口径默认值 — 单一事实源 constants/operating_conditions.json (transient_*)。

    >>> s = load_transient_spec()
    >>> s["steps_per_period"] > 0 and 0.5 < s["theta"] <= 1.0
    True
    """
    oc = _operating()
    return {"theta": float(oc["transient_theta"]),
            "periods": float(oc["transient_periods"]),
            "steps_per_period": int(oc["transient_steps_per_period"]),
            "sigma_magnet": float(oc["transient_sigma_magnet"]),
            "sigma_iron": float(oc["transient_sigma_iron"]),
            "rotor_sweep_steps": int(oc["transient_rotor_sweep_steps"])}


def electrical_freq_hz(speed_rpm, n_poles):
    """电角频率 f = p·n/60 (Hz), p=极对数=n_poles/2。

    >>> electrical_freq_hz(3000, 4)
    100.0
    """
    return (n_poles / 2.0) * speed_rpm / 60.0


def pole_pitch_deg(n_poles):
    """极距角 = 360/n_poles (机械度)。"""
    return 360.0 / n_poles


def time_step_s(period_s, steps_per_period):
    """dt = T/steps (θ 法每电周期步数)。"""
    return period_s / steps_per_period


def check_rotation_step(delta_deg, dtheta_max):
    """剪切带步进角约束: |Δ| ≤ dtheta_max (否则气隙斜单元翻转/退化)。"""
    if abs(delta_deg) > dtheta_max + 1e-12:
        raise ValueError(
            f"转子步进角 {delta_deg}° 超过网格最大角距 {dtheta_max}° "
            f"(剪切带单元翻转风险)")
    return True


def progress_keys():
    """_progress.json 统一键序 (C++ 轮询契约, 与静磁链 ProgressReporter 一致)。"""
    return ["stage", "percent", "elapsed_s", "eta_s", "msg", "timestamp"]


if __name__ == "__main__":
    import doctest
    fails, tested = doctest.testmod().failed, doctest.testmod().attempted
    print(f"transient_spec 口径卡 doctest: {tested - fails}/{tested} PASS")
    raise SystemExit(1 if fails else 0)

#!/usr/bin/env python3
"""M87 场路耦合 (电压源驱动) 口径卡 (doctest 可执行) — 数值口径唯一事实源

链路: run_emag_coupled.py → generate_geo.py(Δ=0) → motor_mag.pro(静磁 Picard,
      逐端口电流点场解) + python 相变量电压方程 → coupled2_report.json

口径 (耦合迭代策略):
- 未知量: a (磁矢位, GetDP 侧) 与 i_ph(t) (端口相电流, 外电路侧)。
- 磁场方程 (GetDP 静磁 Picard, σ=0 端口绕组无涡流):
      ∮ ν(B)∇a·∇δa = ∫ hc·δa − ∫ js(i)·δa ,
  js_ph = 2·Nc·i_ph/A_slot (双层整距, 与静磁链 js_to_i 互逆)。
- 绕组电压方程 (相变量外电路, Y 接无中线, 零序恒零):
      u_ph(t_k) = R_ph(T_w)·i_ph(t_k) + [λ_ph(t_k) − λ_ph(t_{k−1})]/dt ,
  后向 Euler; R_ph 取热耦合终值 (工作点锚, coupled_report.coupled_final)。
- 相磁链 λ_ph: 12 槽相带 (PHASE_BELT), 槽中心带 Az 采样,
      λ_ph = 2·Nc·L·[Σ A(+带槽心) − Σ A(−带槽心)]  (与 flux_linkage 同机制分相)。
- 耦合迭代策略 (v1: 外部磁链迭代分段耦合 — 不动点收敛后磁场方程与电压方程
  同时满足, 等效联立; GetDP Circuit/Network 真联立列为 v2, 需分支电路网格):
  每时间步内层不动点 (磁链迭代 + 端部漏感加速):
      i^{(j+1)} = i^{(j)} − ω·[R·i^{(j)} + (λ(i^{(j)}) − λ_{k−1})/dt − u_k]
                  / (R + L_acc/dt) ,
  L_acc = 1.5·L_cal (标定波弦斜率, 运行时估计, 覆盖互感差模 1.5 倍),
  ω = coupling_relax 欠松弛。
- 收敛判据: max_ph |i^{(j+1)} − i^{(j)}| / i_pk ≤ coupling_itol (1e-3),
  内层步数 ≤ coupling_iter_max (12), 超限记未收敛 (报告披露, 禁静默)。
- 电压源标定 (等效电压反演): 以基线电流波 (M86 口径三相正弦, δ=90° 快照)
  逐点场解得 λ^base_k, 反演
      u_k = R·i^base_k + (λ^base_k − λ^base_{k−1})/dt   (周期延拓, λ_{−1}=λ_{N−1})
  → 电压驱动耦合复现基线电流 → 对齐静磁 load 基线。
- 周期稳态口径 (v1): 电压取周期稳态解, 跳过起动暂态; 电流初值即稳态吸引子。
- 工作点锚: coupled_report.json coupled_final (Hc_Am=841046.1 热磁体, R_ph_ohm
  =41.595 热绕组, T_dl_Nm=−0.8966 记录基线)。对齐锚 = **同网格新鲜静磁 load 解**
  (δ=90° 快照, 同 Hc): 对齐判据考核的是场路耦合对静磁 load 解的复现精度;
  存储值 −0.8966 出自 M85 旧网格 (M86 改 generate_geo 前复算), 新网格静磁复现
  −0.8848 (网格漂移 1.3%, M85 报告过期, 单独披露)。验收: ia=+i_pk 快照步
  Maxwell 转矩 vs 同网格静磁锚 误差 ≤1% (TORQUE_TOL)。

>>> spec = load_coupling_spec()
>>> spec["itol"]
0.001
>>> ang = phase_slot_angles(12)
>>> ang["a"]
([0.0, 180.0], [90.0, 270.0])
>>> ang["b"]
([60.0, 240.0], [150.0, 330.0])
>>> ang["c"]
([120.0, 300.0], [30.0, 210.0])
>>> nc, a_slot = 100, 1e-4
>>> round(js_from_i(1.0, nc, a_slot), 9)
2000000.0
>>> round(i_from_js(2000000.0, nc, a_slot), 9)
1.0
>>> # 电压方程离散残差: u − R·i − Δλ/dt
>>> round(circuit_residual(u=48.0, r=40.0, i=1.0, lam=0.02, lam_prev=0.0, dt=0.0025), 9)
0.0
>>> # 线性磁链 λ(i)=L·i+λm: 一次修正即精确 (L_acc=L); 不动点 64=40i+40i → i*=0.8
>>> i1 = flux_iteration_step(i=0.9, lam_fn=lambda ii: 0.1*ii + 0.5,
...                          u_k=64.0, lam_prev=0.5,
...                          dt=0.0025, r=40.0, l_acc=0.1, relax=1.0)
>>> abs(i1 - 0.8) < 1e-12
True
>>> # 收敛判据
>>> check_converged(di=[1e-4], i_ref=1.0, itol=1e-3)
True
>>> check_converged(di=[2e-3], i_ref=1.0, itol=1e-3)
False
>>> # 弦斜率估计 (差分 → 标定电感)
>>> round(estimate_l_acc(lam=[0.5, 0.6], cur=[1.0, 2.0]), 9)
0.1
>>> round(torque_err(-0.8848, -0.8966), 5)
0.01316
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONST = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "constants")

# 验收判据 (口径卡级, 非业务数值): 稳态转矩对齐容差 = 用户验收线 1%
TORQUE_TOL = 0.01

# 12 槽 4 极双层整距 60° 相带基础序列 (generate_geo.PHASE_BELT 唯一事实源)
PHASE_BELT = ["A+", "C-", "B+", "A-", "C+", "B-"]


def _operating():
    with open(os.path.join(_CONST, "operating_conditions.json"),
              encoding="utf-8") as f:
        return json.load(f)


def load_coupling_spec():
    """耦合口径默认值 — 单一事实源 constants/operating_conditions.json (coupling_*)。

    >>> s = load_coupling_spec()
    >>> 0 < s["itol"] < 0.1 and s["iter_max"] > 0 and 0 < s["relax"] <= 1.0
    True
    """
    oc = _operating()
    return {"itol": float(oc["coupling_itol"]),
            "iter_max": int(oc["coupling_iter_max"]),
            "relax": float(oc["coupling_relax"])}


def phase_slot_angles(n_slots=12):
    """相带 → (正带槽心角列表, 负带槽心角列表) [机械度]。序列按 PHASE_BELT 循环。

    >>> phase_slot_angles(24)["a"][0]
    [0.0, 90.0, 180.0, 270.0]
    """
    if n_slots % 6 != 0:
        raise ValueError(f"n_slots={n_slots} 非6的倍数, 分数槽绕组暂不支持")
    pitch = 360.0 / n_slots
    out = {}
    for ph in "abc":
        plus = [k * pitch for k in range(n_slots)
                if PHASE_BELT[k % 6] == ph.upper() + "+"]
        minus = [k * pitch for k in range(n_slots)
                 if PHASE_BELT[k % 6] == ph.upper() + "-"]
        out[ph] = (plus, minus)
    return out


def js_from_i(cur, nc, a_slot):
    """相电流(A) → 槽电流密度(A/m²): js = 2·Nc·i/A_slot (双层)。"""
    return 2.0 * nc * cur / a_slot


def i_from_js(js, nc, a_slot):
    """槽电流密度(A/m²) → 相电流(A) (js_to_i 同口径)。"""
    return js * a_slot / (2.0 * nc)


def circuit_residual(u, r, i, lam, lam_prev, dt):
    """电压方程离散残差: u − R·i − (λ − λ_prev)/dt (后向 Euler)。"""
    return u - r * i - (lam - lam_prev) / dt


def flux_iteration_step(i, lam_fn, u_k, lam_prev, dt, r, l_acc, relax):
    """内层磁链迭代一步: 电流修正 (端部漏感加速分母 R + L_acc/dt)。

    lam_fn: i → λ (当前三相电流分布下场解的该相磁链)。"""
    lam = lam_fn(i)
    res = circuit_residual(u_k, r, i, lam, lam_prev, dt)
    # res 对 i 递减 (d res/di = −(R + L_real/dt) < 0): res>0 → 需增大 i 压回
    return i + relax * res / (r + l_acc / dt)


def check_converged(di, i_ref, itol):
    """收敛判据: max|Δi| / i_ref ≤ itol。"""
    return max(abs(x) for x in di) <= itol * abs(i_ref)


def estimate_l_acc(lam, cur):
    """标定弦斜率 L_cal = Δλ/Δi (差分), 供 L_acc = 1.5·L_cal。

    >>> round(estimate_l_acc(lam=[0.0, 0.3], cur=[0.0, 3.0]), 9)
    0.1
    """
    if len(lam) < 2:
        raise ValueError("标定样本不足 (需 ≥2 点)")
    num = lam[-1] - lam[0]
    den = cur[-1] - cur[0]
    if den == 0.0:
        raise ValueError("标定电流无差分 (Δi=0), 弦斜率退化")
    return num / den


def torque_err(t, t_ref):
    """稳态转矩对齐误差 |T − T_ref| / |T_ref|。"""
    return abs(t - t_ref) / abs(t_ref)


if __name__ == "__main__":
    import doctest
    fails, tested = doctest.testmod().failed, doctest.testmod().attempted
    print(f"coupling_spec 口径卡 doctest: {tested - fails}/{tested} PASS")
    raise SystemExit(1 if fails else 0)

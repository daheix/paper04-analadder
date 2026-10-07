#!/usr/bin/env python3
"""M100 温变磁体激励 口径卡 (doctest 可执行) — 数值口径唯一事实源

链路: LPTN 磁体节点温度 (run_thermal_getdp.py → thermal_report.T_magnet, 单向)
      → 电磁链激励重算 (run_emag_getdp.py / run_emag_transient.py):
      Hc(T) = Hc0·(1+α_Br·(T−20)), Br(T) = Br20·(1+α_Br·(T−20))

口径:
- 线性温变模型 (一阶可逆温度系数):
      Br(T) = Br20·(1 + α_Br·(T−20))   [T]
      Hc(T) = Hc0·(1 + α_Br·(T−20))   [A/m]  (μ_r 不随温变, 磁体本构 B=μ_r·μ0·H+Br)
- 磁链 λm(T): 磁链对磁体磁通线性 → λm(T) = λm20·(1+α_Br·(T−20))。
- 转矩 T(T) = T20·(1+α_Br·(T−20)) (T = 1.5·p·λm·iq, iq 不随磁体温变)。
- 反电势 E(T) = ω·λm(T) → 同一缩放因子 k(T) = 1+α_Br·(T−20)。
- 敏感度 (线性口径): ΔBg1/Bg1 = ΔT_torque/T = α_Br·ΔT (解析, 无需重解);
  报告披露该值为线性近似 — 对标 arXiv:2410.16240 含饱和+温度非线性磁模型,
  本链未计饱和-温度交叉项 (单向披露, 禁静默)。
- 缺省 α_Br=0 (constants/materials.json 无该字段时) → 任意温度 k=1, 行为向后兼容。
- 温度来源唯一: 设计键 T_Magnet_C (LPTN 磁体节点, 单向); 缺省 20°C=常温不变。
- 锚点 (N35UH: Br20=1.16 T, Hc0=880400 A/m, α_Br=−0.0012 /K):
      T=120°C: k=0.88, Br=1.0208 T, Hc=774752.0 A/m, λm=0.9798→0.862224 Wb
      T=60°C:  k=0.952, Br=1.10432 T
- doctest 覆盖: 3 种 α_Br (−0.0012/−0.0005/0.0) × 2 温度点 (60/120°C)。

>>> spec = load_temp_spec()
>>> spec["t_ref_C"]
20.0
>>> spec["anchors"]["N35UH"]["alpha_br_per_K"]
-0.0012
>>> k_br(-0.0012, 120.0)
0.88
>>> k_br(-0.0012, 60.0)
0.952
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_temp_spec():
    """读本口径卡锚点 (constants 单一事实源约定: 锚点随卡, 不散落)。

    >>> s = load_temp_spec()
    >>> s["anchors"]["N35UH"]["br20_T"]
    1.16
    >>> s["anchors"]["N35UH"]["hc0_Am"]
    880400.0
    >>> s["sensitivity_dT_K"]
    50.0
    """
    with open(os.path.join(HERE, "temp_spec.json"), encoding="utf-8") as f:
        return json.load(f)


def alpha_br_of(mat):
    """材质条目 → α_Br (/K); 缺省 0=常温不变 (向后兼容)。

    >>> alpha_br_of({"ALPHA_BR": -0.0012})
    -0.0012
    >>> alpha_br_of({"Br_20": 1.16})
    0.0
    >>> alpha_br_of({})
    0.0
    """
    return float(mat.get("ALPHA_BR", 0.0))


def k_br(alpha_br, t, t_ref=20.0):
    """线性温变缩放因子 k(T) = 1 + α_Br·(T−T_ref)。

    3 种 α_Br × 2 温度点 (60/120°C) 覆盖:
    >>> k_br(-0.0012, 120.0)
    0.88
    >>> k_br(-0.0012, 60.0)
    0.952
    >>> k_br(-0.0005, 120.0)
    0.95
    >>> k_br(-0.0005, 60.0)
    0.98
    >>> k_br(0.0, 120.0)
    1.0
    >>> k_br(0.0, 60.0)
    1.0
    """
    return 1.0 + float(alpha_br) * (float(t) - float(t_ref))


def br_at_t(br20, alpha_br, t):
    """Br(T) = Br20·k(T) [T] (N35UH 锚点)。

    >>> br_at_t(1.16, -0.0012, 120.0)
    1.0208
    >>> br_at_t(1.16, -0.0012, 60.0)
    1.10432
    >>> br_at_t(1.16, -0.0005, 120.0)
    1.102
    >>> br_at_t(1.16, -0.0005, 60.0)
    1.1368
    >>> br_at_t(1.16, 0.0, 120.0)
    1.16
    >>> br_at_t(1.16, 0.0, 60.0)
    1.16
    """
    return round(float(br20) * k_br(alpha_br, t), 9)


def hc_at_t(hc0, alpha_br, t):
    """Hc(T) = Hc0·k(T) [A/m] — GetDP 激励注入值 (-setnumber Hc_mag)。

    >>> hc_at_t(880400.0, -0.0012, 120.0)
    774752.0
    >>> hc_at_t(880400.0, -0.0012, 60.0)
    838140.8
    >>> hc_at_t(880400.0, -0.0005, 120.0)
    836380.0
    >>> hc_at_t(880400.0, 0.0, 120.0)
    880400.0
    """
    return round(float(hc0) * k_br(alpha_br, t), 6)


def lambda_m_at_t(lam20, alpha_br, t):
    """相磁链 λm(T) = λm20·k(T) [Wb] (锚 λm20=0.9798, M85 基线)。

    >>> lambda_m_at_t(0.9798, -0.0012, 120.0)
    0.862224
    >>> lambda_m_at_t(0.9798, -0.0012, 60.0)
    0.93277
    >>> lambda_m_at_t(0.9798, 0.0, 120.0)
    0.9798
    """
    return round(float(lam20) * k_br(alpha_br, t), 6)


def torque_at_t(t20, alpha_br, t):
    """转矩 T(T) = T20·k(T) [Nm] (锚 T20=−0.8966, M85 负载基线)。

    >>> torque_at_t(-0.8966, -0.0012, 120.0)
    -0.789008
    >>> torque_at_t(-0.8966, -0.0005, 120.0)
    -0.85177
    >>> torque_at_t(-0.8966, 0.0, 60.0)
    -0.8966
    """
    return round(float(t20) * k_br(alpha_br, t), 6)


def bemf_at_t(e20, alpha_br, t):
    """反电势 E(T) = ω·λm(T) → E20·k(T) [V] (同一线性缩放)。

    >>> bemf_at_t(100.0, -0.0012, 120.0)
    88.0
    >>> bemf_at_t(100.0, -0.0005, 60.0)
    98.0
    """
    return round(float(e20) * k_br(alpha_br, t), 6)


def sensitivity_pct(alpha_br, d_t=50.0):
    """线性敏感度: Δ量/量 = α_Br·ΔT (Bg1 与转矩同因子) [%]。

    >>> sensitivity_pct(-0.0012, 50.0)
    -6.0
    >>> sensitivity_pct(-0.0005, 50.0)
    -2.5
    >>> sensitivity_pct(0.0, 50.0)
    0.0
    """
    return round(float(alpha_br) * float(d_t) * 100.0, 6)


if __name__ == "__main__":
    import doctest
    fails, tested = doctest.testmod().failed, doctest.testmod().attempted
    print(f"temp_spec 口径卡 doctest: {tested - fails}/{tested} PASS")
    raise SystemExit(1 if fails else 0)

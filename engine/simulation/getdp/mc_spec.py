#!/usr/bin/env python3
"""M94 公差蒙特卡洛口径卡 (doctest 可执行) — 对标 ANSYS PExprt/SPEED 公差分析

链路: mc_spec.py (本卡) + run_mc_tolerance.py (驱动器)
      + constants/mc_tolerances.json (mc_* 白名单)
      → LHS 拉丁超立方抽样 N≥200 (每维分层+随机置换, 每层恰 1 样本)
      → 逐样本解析快速估算 (复用 M92 概念流正向公式, 不跑全 FEM)
      → mc_report.json (P50/P95/σ/散布带 + 敏感性排序 + 收敛判据)

口径 (拉丁超立方 LHS):
    每维 [0,1) 均分 n 层, 第 i 行在第 i 层内均匀抽样 u=(i+ξ)/n, ξ~U(0,1),
    维间独立随机置换 → 边缘分布满覆盖 (优于纯随机, 同 N 方差更小);
    逆变换采样: uniform → lo+u·(hi−lo); normal → mean+σ·Φ⁻¹(u)。

口径 (解析快速估算 — M92 概念流正向公式, 全部披露):
    气隙 g = R_si−(R_ri+mag_t); 磁路 Bg = Br·hm/(hm+μr·g) (bg_from_mag_circuit);
    偏心平均折减 f_ecc = 1/(1+k_ecc·(e/g)²) (二阶平均效应, 披露: 不含单边磁拉力);
    每极磁通 λ1=(2/π)·Bg_eff·τ_p·L; 磁链 λ_m=kw·N·λ1;
    Iq=T/(1.5·p·λ_m); I_rms=Iq/√2;
    ρ(T)=ρ20·(1+α(T−20)); R_ph=ρ(T)·L_turn·N/A_wire (r_dc_winding);
    P_cu=3·I_rms²·R_ph; Bertotti/Steinmetz P_fe=(KH·f+KE·f²)·Bg_eff²·V_fe·k_manuf;
    η=P_out/(P_out+P_cu+P_fe)。

口径 (统计与收敛):
    指标: mean/σ/P05/P50/P95, 散布带 = P95−P05;
    标称两口径 (均披露):
      ①中值标称 = 公差带中值正向 (名义制造中心) — 含 Jensen 偏差 E[f(X)]≠f(E[X]),
        只作工程参考, 不作收敛判据;
      ②分布均值参考 = 大样本解析复制 (N_ref=20000, 同 LHS 口径) — 收敛判据基准;
    收敛判据: |mean_N − mean_ref| ≤ k·σ/√N (k=2, mc_convergence_k) —
      中心极限 2σ 口径 (采样误差随 N 收敛, 模型非线性偏差不随 N 收敛故排除出判据);
    敏感性: |Spearman 秩相关| |样本键值 vs 输出指标|, 降序排序。

口径 (工艺铁损修正 k_manuf, 默认关):
    Bertotti 三分量 (kh·f+ke·f²)·B² 乘 k_manuf (冲剪硬化区铁损增量系数,
    materials.manufacturing.k_manuf, 默认 1.15); k_manuf_enabled=false → ×1.0 (理想口径)。

>>> tol = load_mc()
>>> tol["mc_n_samples"]
200
>>> tol["mc_convergence_k"]
2.0
>>> # LHS: n 行每维严格分层 (每层恰 1 样本) + 置换后仍满覆盖
>>> s = lhs_unit(8, 2, random.Random(7))
>>> len(s), len(s[0])
(8, 2)
>>> all(sum(1 for r in s if i / 8 <= r[j] < (i + 1) / 8) == 1 for j in (0, 1) for i in range(8))
True
>>> # 逆变换: uniform 线性; 样本落界内
>>> round(sample_value({"dist": "uniform", "lo": 0.0, "hi": 0.3}, 0.5), 6)
0.15
>>> round(sample_value({"dist": "normal", "mean": 10.0, "sigma": 2.0}, 0.5), 6)
10.0
>>> # 磁路+偏心折减: 基线锚 Bg=0.859259 (hm=3mm,g=1mm); e=0 → 不折减
>>> round(bg_eff(1.16, 0.003, 0.001, 1.05, 0.0, 0.5), 6)
0.859259
>>> e_half = bg_eff(1.16, 0.003, 0.001, 1.05, 0.0005, 0.5)
>>> 0 < e_half < 0.859259
True
>>> # 电阻温度: ρ(T)=ρ20(1+α(T−20)), 100°C vs 20°C 比值=1+0.00393·80
>>> round(rho_cu_temp(1.68e-8, 100.0) / 1.68e-8, 6)
1.3144
>>> # 工艺修正: k_manuf 开关 — 关=×1.0 (即使传 k_manuf 也无效), 开=×k
>>> round(iron_loss_bertotti(150.0, 0.05, 100.0, 1.0, 1e-4, 1.0), 6)
1.55
>>> round(iron_loss_bertotti(150.0, 0.05, 100.0, 1.0, 1e-4, 1.15), 6)
1.55
>>> round(iron_loss_bertotti(150.0, 0.05, 100.0, 1.0, 1e-4, 1.0, enabled=False), 6)
1.55
>>> round(iron_loss_bertotti(150.0, 0.05, 100.0, 1.0, 1e-4, 1.15, enabled=True), 6)
1.7825
>>> # 统计: 确定数列的 P50/σ/散布带 (线性插值百分位)
>>> sm = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
>>> round(sm["mean"], 6), round(sm["sigma"], 6), sm["p50"], sm["p05"], sm["p95"]
(3.0, 1.581139, 3.0, 1.2, 4.8)
>>> round(sm["spread_band"], 6)
3.6
>>> # 收敛判据: |mean−nominal| ≤ k·σ/√N (2σ/√5=1.414)
>>> convergence_ok({"mean": 3.0}, 3.0, 1.581139, 5, 2.0)
True
>>> convergence_ok({"mean": 3.0}, 2.5, 1.581139, 5, 2.0)
True
>>> convergence_ok({"mean": 3.0}, 1.0, 1.581139, 5, 2.0)
False
>>> # 敏感性: 完全单调相关=±1, 无关=0
>>> round(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 6)
1.0
>>> round(spearman([1, 2, 3, 4], [40, 30, 20, 10]), 6)
-1.0
>>> round(spearman([1, 2, 3, 4], [1, 1, 1, 1]), 6)
0.0
>>> # 标称样本 = 公差带中值 (名义制造中心)
>>> mc2 = {"mc_tolerances": {"mag_t_mm": {"dist": "uniform", "lo": -0.05, "hi": 0.05},
...                          "eccentricity_mm": {"dist": "uniform", "lo": 0.0, "hi": 0.3}},
...        "mc_winding_temp_c": {"dist": "uniform", "lo": 20.0, "hi": 100.0}}
>>> ns = nominal_sample(mc2)
>>> ns == {"mag_t_mm": 0.0, "eccentricity_mm": 0.15, "winding_temp_c": 60.0}
True
>>> # 非法输入 fail-fast
>>> lhs_unit(0, 2, random.Random(1))
Traceback (most recent call last):
    ...
ValueError: 样本数/维度必须为正: n=0, dim=2
"""
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import load, materials as _mat_fn, physics as _phy_fn
from concept_spec import bg_from_mag_circuit
from ac_loss_spec import r_dc_winding

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def load_mc():
    """mc_* 白名单 (fail-fast)。"""
    return load("mc_tolerances.json")


def lhs_unit(n, dim, rng):
    """拉丁超立方抽样 → n×dim 列表, 每维严格分层 (每层恰 1 样本), 维间独立置换。"""
    if n <= 0 or dim <= 0:
        raise ValueError(f"样本数/维度必须为正: n={n}, dim={dim}")
    cols = []
    for _ in range(dim):
        col = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(col)
        cols.append(col)
    return [[cols[j][i] for j in range(dim)] for i in range(n)]


def sample_value(spec, u):
    """逆变换采样: uniform→lo+u·(hi−lo); normal→mean+σ·Φ⁻¹(u)。"""
    d = spec["dist"]
    if d == "uniform":
        return spec["lo"] + u * (spec["hi"] - spec["lo"])
    if d == "normal":
        return spec["mean"] + spec["sigma"] * statistics.NormalDist().inv_cdf(u)
    raise ValueError(f"未知分布: {d}")


def bg_eff(br, h_m, gap, mur, ecc_m, k_ecc):
    """偏心折减磁密 Bg_eff = Br·hm/(hm+μr·g)/(1+k_ecc·(e/g)²) (二阶平均, 披露)。"""
    bg = bg_from_mag_circuit(br, h_m, gap, mur)
    if ecc_m < 0:
        raise ValueError(f"偏心量必须非负: {ecc_m}")
    return bg / (1.0 + k_ecc * (ecc_m / gap) ** 2)


def rho_cu_temp(rho_20, temp_c):
    """铜电阻率温度修正 ρ(T)=ρ20·(1+α·(T−20)), α=physics.ALPHA_CU。"""
    return rho_20 * (1.0 + _phy_fn()["ALPHA_CU"] * (temp_c - 20.0))


def iron_loss_bertotti(kh, ke, freq, b, v_fe, k_manuf, enabled=False):
    """Bertotti/Steinmetz 铁损 P_fe=(KH·f+KE·f²)·B²·V_fe × k_manuf 开关 (默认关=×1.0)。"""
    if freq < 0 or b < 0 or v_fe < 0:
        raise ValueError(f"非法铁损输入: f={freq}, B={b}, V={v_fe}")
    p = (kh * freq + ke * freq * freq) * b * b * v_fe
    if enabled:
        if k_manuf < 1.0:
            raise ValueError(f"工艺增量系数必须≥1: {k_manuf}")
        p *= k_manuf
    return p


def summarize(values):
    """指标束: mean/σ/P05/P50/P95/散布带 (P95−P05)。"""
    if not values:
        raise ValueError("空样本列")
    vs = sorted(values)
    n = len(vs)

    def pct(p):
        k = p * (n - 1)
        f = int(math.floor(k))
        c = min(f + 1, n - 1)
        return vs[f] + (k - f) * (vs[c] - vs[f])

    mean = statistics.fmean(vs)
    sigma = statistics.stdev(vs) if n >= 2 else 0.0
    p05, p95 = pct(0.05), pct(0.95)
    return {"mean": mean, "sigma": sigma, "p05": p05, "p50": pct(0.50),
            "p95": p95, "spread_band": p95 - p05}


def convergence_ok(summary, nominal, sigma, n, k):
    """收敛判据: |mean−nominal| ≤ k·σ/√N (中心极限 2σ 口径)。"""
    if n <= 1 or k <= 0:
        raise ValueError(f"非法收敛判据输入: N={n}, k={k}")
    return abs(summary["mean"] - nominal) <= k * sigma / math.sqrt(n)


def _ranks(xs):
    """平均秩 (并列取平均) — Spearman 口径。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman 秩相关 ρ (平均秩并列口径); 常数列→0。"""
    if len(xs) != len(ys) or not xs:
        raise ValueError("秩相关输入长度不一致或为空")
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def nominal_sample(mc):
    """标称样本 = 公差带中值 (名义制造中心): uniform→(lo+hi)/2, normal→mean。

    收敛判据口径: MC 均值 vs 中值标称的偏差 ≤ k·σ/√N (排偏心单边带的中值偏置)。
    """
    nom = {}
    for k, spec in mc["mc_tolerances"].items():
        if k.startswith("_"):
            continue
        nom[k] = (spec["mean"] if spec["dist"] == "normal"
                  else 0.5 * (spec["lo"] + spec["hi"]))
    t = mc["mc_winding_temp_c"]
    nom["winding_temp_c"] = (t["mean"] if t["dist"] == "normal"
                             else 0.5 * (t["lo"] + t["hi"]))
    return nom


def forward_analytic(sample, ctx):
    """单样本解析快速估算 (M92 概念流正向公式) → 输出指标 dict。

    ctx: 预展开的标称上下文 (D/L/τp/V_fe/A_wire/KH/KE/Br/μr/工况), 逐样本只
    重算受公差键影响的量 (Bg/λm/I/R/P_cu/P_fe/η), 几何定尺按标称固定 (披露)。
    """
    g = ctx["gap_nom_m"] + sample["airgap_mm"] * 1e-3
    if g <= 0:
        raise ValueError(f"气隙必须为正: {g}")
    hm = ctx["hm_nom_m"] + sample["mag_t_mm"] * 1e-3
    if hm <= 0:
        raise ValueError(f"磁厚必须为正: {hm}")
    n_turns = ctx["N_nom"] + sample["Nc_turns"]
    ecc_m = sample["eccentricity_mm"] * 1e-3
    b = bg_eff(ctx["Br"], hm, g, ctx["mur"], ecc_m, ctx["k_ecc"])
    lam1 = (2.0 / math.pi) * b * ctx["tau_p_m"] * ctx["L_m"]
    lam_m = ctx["kw"] * n_turns * lam1
    iq = ctx["torque_Nm"] / (1.5 * ctx["p"] * lam_m)
    i_rms = iq * _INV_SQRT2
    rho_t = rho_cu_temp(ctx["rho20"], sample["winding_temp_c"])
    r_ph = r_dc_winding(rho_t, ctx["L_turn_m"], n_turns, ctx["A_wire_m2"])
    p_cu = 3.0 * i_rms * i_rms * r_ph
    f_elec = ctx["p"] * ctx["speed_rpm"] / 60.0
    p_fe = iron_loss_bertotti(ctx["KH"], ctx["KE"], f_elec, b, ctx["V_fe_m3"],
                              ctx["k_manuf"], enabled=ctx["k_manuf_enabled"])
    p_out = ctx["torque_Nm"] * (2.0 * math.pi * ctx["speed_rpm"] / 60.0)
    eta = p_out / (p_out + p_cu + p_fe)
    return {"Bg1_T": b, "lam_m_Wb": lam_m, "I_rms_A": i_rms,
            "P_cu_W": p_cu, "P_fe_W": p_fe, "eta_pct": eta * 100.0}


def build_context(manuf_enabled=False):
    """预展开标称上下文 (M92 概念流定尺一次, MC 逐样本只重算扰动量)。"""
    from load_constants import default_design, operating
    from concept_spec import concept_design

    mc = load_mc()
    concept = load("concept_design.json")
    dflt, op = default_design(), operating()
    wpt = mc["mc_operating_point"]
    # 标称定尺一次 (D/L/τp/A_wire 取 M92 概念流标称口径, 公差只扰磁/电/温度链)
    rep = concept_design(target_torque=wpt["torque_Nm"], speed_rpm=wpt["speed_rpm"],
                         voltage=wpt["voltage_V"], p=wpt["p"])
    d = rep["sizing"]["D_mm"] * 1e-3
    length = rep["sizing"]["L_mm"] * 1e-3
    tau_p = rep["sizing"]["tau_p_mm"] * 1e-3
    est = rep["estimates"]
    mat = _mat_fn()
    steel = mat["steel_50JN350"]
    manuf = mat["manufacturing"]
    d2l0 = dflt["R_so"] ** 2 * dflt["L_stack"]
    i_rms_nom = est["I_rms_A"]
    return {
        "gap_nom_m": dflt["R_si"] - (dflt["R_ri"] + dflt["mag_t"]),
        "hm_nom_m": dflt["mag_t"],
        "N_nom": dflt["Nc"],
        "Br": mat["magnet_N35UH"]["Br_20"],
        "mur": mat["magnet_N35UH"]["mur"],
        "rho20": mat["copper"]["RHO_20"],
        "kw": rep["inputs"]["kw"],
        "k_ecc": mc["mc_ecc_coeff"],
        "torque_Nm": wpt["torque_Nm"], "speed_rpm": wpt["speed_rpm"], "p": wpt["p"],
        "D_m": d, "L_m": length, "tau_p_m": tau_p,
        "L_turn_m": 2.0 * (length + concept["concept_end_turn_tau_frac"] * tau_p),
        "A_wire_m2": i_rms_nom / concept["concept_current_density_A_m2"],
        "V_fe_m3": op["V_fe_m3"] * (d * d * length) / d2l0,
        "KH": steel["KH"], "KE": steel["KE"],
        "k_manuf": manuf["k_manuf"],
        "k_manuf_enabled": bool(manuf_enabled),
    }


def build_samples(mc, seed=None):
    """LHS 全样本: 键序=mc_tolerances 白名单序 + winding_temp_c → 值矩阵。"""
    keys = list(mc["mc_tolerances"].keys())
    keys = [k for k in keys if not k.startswith("_")] + ["winding_temp_c"]
    rng = random.Random(seed if seed is not None else mc["mc_seed"])
    rows = lhs_unit(mc["mc_n_samples"], len(keys), rng)
    return keys, [{k: sample_value(mc["mc_tolerances"].get(k, mc["mc_winding_temp_c"]), u[j])
                   for j, k in enumerate(keys)} for u in rows]


def sensitivity(samples, outputs, keys, out_keys):
    """敏感性排序: 每输出按 |Spearman(键, 输出)| 降序。"""
    ranking = {}
    for ok in out_keys:
        ys = [o[ok] for o in outputs]
        scs = [(k, abs(spearman([s[k] for s in samples], ys))) for k in keys]
        scs.sort(key=lambda t: -t[1])
        ranking[ok] = [{"key": k, "spearman_abs": round(v, 4)} for k, v in scs]
    return ranking


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"mc_spec doctest: {r.attempted} 例, 失败 {r.failed}")
    raise SystemExit(1 if r.failed else 0)

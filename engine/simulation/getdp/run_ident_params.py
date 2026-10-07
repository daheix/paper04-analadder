#!/usr/bin/env python3
"""M95 参数辨识驱动器 (数字孪生雏形) — 对标 SPEED/Motor-CAD 参数校准

链路: constants/opt_design.json (ident_* 白名单) + opt_spec.py (口径卡)
      → "实测" = 标定库代表模型 (ident_model=baseline-12s4p) reference_metrics
        (Bg1=0.9155T / λm=0.9798Wb / T=0.8966Nm, 名义 FEM 逐位复现已验证)
      → scipy least_squares (trf) 辨识 {Br, kw} 物理界约束
      → ident_report.json (初值/辨识值/残差范数/相对误差/灵敏度矩阵条件数/可辨识性)

k_ecc 口径 (全披露, 禁静默兜底):
    设计意图 f_ecc=1/(1+k_ecc·(e/g)²) (M94 二阶平均折减); 但本链偏心 FEM 实测
    (e=0.5mm/g=1mm) 基波 Bg1 +1.17%、极均值 Bg_pole_mean +15.4% — 磁路
    Bg∝1/(hm+μr·g(θ)) 为凸函数, 偏心使平均磁密增大 (Jensen 不等式), 折减模型
    方向与 FEM 相反 → k_ecc 按折减口径不可辨识 (隐含 k<0 越物理界), 报告披露
    根因与实测数字, 不入拟合; k_ecc 精确辨识需单边局部磁密口径 (后续工作)。

模型口径 (M92 磁路同式):
    Bg1(Br) = Br·hm/(hm+μr·g)·k_f; λm(Br,kw) = kw·N_ph·(2/π)·Bg1·τp·L;
    T(Br,kw) = 1.5·p·λm·Iq, Iq = js·A_slot/(2·Nc) (js_to_i 同式, js=ident_rated_js_A_m2)。
残差 = (模型−实测)/实测 (相对误差); 灵敏度矩阵条件数 ≤ ident_cond_max 披露,
列范数近零的参数披露为弱可辨识; 残差 |相对误差| ≤ ident_res_tol_rel 验收全披露。

用法: python3 run_ident_params.py   (在 simulation/getdp/ 目录下执行; 偏心 FEM 实测
      缺失时自动跑 run_emag_getdp.py --no-sweep --params --workdir 绝对路径隔离)
输出: ident_report.json
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import default_design, materials
from mc_spec import bg_eff
from opt_spec import (check_ident_bounds, ecc_factor, load_opt, pole_arc_coef,
                      sens_cond, slot_area)


def ident_context(opt):
    """辨识上下文: 设计/材料定尺一次 (基线设计), 辨识只扰 {Br, kw}。"""
    d = default_design()
    mat = materials()
    g = d["R_si"] - d["R_ri"] - d["mag_t"]
    if g <= 0:
        raise ValueError(f"气隙必须为正: {g}")
    alpha = pole_arc_coef(d["mag_half_deg"], d["p"])
    kf = 4.0 / math.pi * math.sin(alpha * math.pi / 2.0)
    r_gap_mid = 0.5 * (d["R_si"] + d["R_ri"] + d["mag_t"])
    return {
        "mag_t": d["mag_t"], "gap_m": g, "mur": mat["magnet_N35UH"]["mur"],
        "k_f": kf, "tau_p": math.pi * r_gap_mid / d["p"], "L": d["L_stack"],
        "N_ph": (d["n_slots"] // 3) * d["Nc"], "p": d["p"], "Nc": d["Nc"],
        "A_slot": slot_area(d["R_si"], d["slot_depth"], d["slot_half_deg"]),
        "js": opt["ident_rated_js_A_m2"], "ecc_m": opt["ident_ecc_mm"] * 1e-3,
    }


def forward_ident(x, ctx):
    """辨识正向模型 → [Bg1_nom, lam_m, T] (x = [Br, kw])。"""
    br, kw = float(x[0]), float(x[1])
    b1 = bg_eff(br, ctx["mag_t"], ctx["gap_m"], ctx["mur"], 0.0, 1.0) * ctx["k_f"]
    lam1 = (2.0 / math.pi) * b1 * ctx["tau_p"] * ctx["L"]
    lam = kw * ctx["N_ph"] * lam1
    i_pk = ctx["js"] * ctx["A_slot"] / (2.0 * ctx["Nc"])
    return [b1, lam, 1.5 * ctx["p"] * lam * i_pk]


def residuals(x, ctx, y):
    """相对残差 r = (模型−实测)/实测, 3 观测 × {Br, kw}。"""
    return [(m - v) / v for m, v in zip(forward_ident(x, ctx), y)]


def _fem_run(design, wd, tag):
    """全 FEM 复核 (--no-sweep --params --workdir 绝对路径隔离, 串行)。"""
    rep_path = os.path.join(wd, "emag_report.json")
    if os.path.exists(rep_path):
        return rep_path
    os.makedirs(wd, exist_ok=True)
    pf = os.path.join(wd, "params.json")
    json.dump(design, open(pf, "w", encoding="utf-8"), indent=1)
    cmd = [sys.executable, "run_emag_getdp.py", "--no-sweep", "--params", pf,
           "--workdir", wd]
    print(f"[辨识] FEM 实测启动 ({tag}) → {wd}")
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    open(os.path.join(wd, "run.log"), "w", encoding="utf-8").write(
        r.stdout + "\n===STDERR===\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"FEM 实测 {tag} 失败 (exit={r.returncode}), "
                           f"日志 {os.path.join(wd, 'run.log')}: "
                           f"{(r.stderr or r.stdout)[-500:]}")
    return rep_path


def fem_ecc_evidence(opt):
    """偏心点 FEM 实测 (基线设计 + e=ident_ecc_mm) + 名义点同口径对照。"""
    import copy
    base = json.load(open(os.path.join(
        HERE, "..", "..", "models", "library", f"{opt['ident_model']}.json"),
        encoding="utf-8"))
    d_ecc = copy.deepcopy(base["design"])
    d_ecc["eccentricity_mm"] = opt["ident_ecc_mm"]
    rep_ecc = json.load(open(_fem_run(
        d_ecc, os.path.join(HERE, "m95_runs", "ident_ecc"), "偏心点"), encoding="utf-8"))
    rep_nom = json.load(open(_fem_run(
        base["design"], os.path.join(HERE, "m95_runs", "ident_nom"), "名义点"),
        encoding="utf-8"))
    e = {"eccentricity_mm": opt["ident_ecc_mm"],
         "Bg1_ecc_T": rep_ecc["no_load"]["Bg1"],
         "Bg1_nom_T": rep_nom["no_load"]["Bg1"],
         "pole_mean_ecc_T": rep_ecc["no_load"]["Bg_pole_mean"],
         "pole_mean_nom_T": rep_nom["no_load"]["Bg_pole_mean"],
         "lambda_m_ecc_Wb": rep_ecc["flux_linkage"]["lambda_m_Wb"]}
    e["fund_pct"] = round((e["Bg1_ecc_T"] / e["Bg1_nom_T"] - 1.0) * 100, 3)
    e["mean_pct"] = round((e["pole_mean_ecc_T"] / e["pole_mean_nom_T"] - 1.0) * 100, 3)
    return e


def k_ecc_implied(evidence, ctx):
    """由极均值实测比反解隐含 k_ecc: f_meas=mean_ecc/mean_nom=1/(1+k(e/g)²)。"""
    f = evidence["pole_mean_ecc_T"] / evidence["pole_mean_nom_T"]
    return (1.0 / f - 1.0) / (evidence["eccentricity_mm"] * 1e-3 / ctx["gap_m"]) ** 2


def sensitivity_matrix(x, ctx, y, eps=1e-6):
    """灵敏度矩阵 S[i][j] = ∂r_i/∂(x_j/x_j,init) (相对量纲, 识别点前向差分)。"""
    r0 = np.array(residuals(x, ctx, y))
    cols = []
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += eps * abs(x[j])
        cols.append((np.array(residuals(xp, ctx, y)) - r0) / eps * abs(x[j]))
    return np.column_stack(cols)


def main():
    opt = load_opt()
    check_ident_bounds(opt["ident_params"])
    names = ["Br", "kw"]
    lo = [opt["ident_params"][k]["lo"] for k in names]
    hi = [opt["ident_params"][k]["hi"] for k in names]
    x0 = np.array([opt["ident_params"][k]["init"] for k in names])

    ctx = ident_context(opt)
    tgt = opt["ident_targets"]
    y = [tgt["Bg1_T"], tgt["lambda_m_Wb"], tgt["torque_Nm"]]
    y_names = ["Bg1_T", "lambda_m_Wb", "torque_Nm"]

    ecc = fem_ecc_evidence(opt)
    k_imp = k_ecc_implied(ecc, ctx)
    k_lo = opt["ident_params"]["k_ecc"]["lo"]
    k_identifiable = bool(k_imp >= k_lo)
    print(f"[辨识] 实测目标: {dict(zip(y_names, [round(v, 5) for v in y]))}")
    print(f"[辨识] 偏心 FEM 证据: 基波 {ecc['fund_pct']:+.2f}%, 极均值 "
          f"{ecc['mean_pct']:+.2f}% → 隐含 k_ecc={k_imp:.4f} "
          f"(界 lo={k_lo}) → 折减口径可辨识: {k_identifiable}")

    fun = lambda x: residuals(x, ctx, y)
    r0 = np.array(fun(x0))
    res = least_squares(fun, x0, bounds=(lo, hi), method="trf")
    x_id, r_id = res.x, np.array(fun(res.x))

    S = sensitivity_matrix(x_id, ctx, y)
    cond = sens_cond(S)
    col_norms = {k: float(np.linalg.norm(S[:, j])) for j, k in enumerate(names)}
    tol = opt["ident_res_tol_rel"]
    pass_all = bool(np.all(np.abs(r_id) <= tol))
    cond_ok = bool(cond <= opt["ident_cond_max"])

    report = {
        "module": "M95", "tool": "scipy least_squares (trf)",
        "ident_model": opt["ident_model"],
        "params": {"names": names,
                   "init": {k: float(v) for k, v in zip(names, x0)},
                   "identified": {k: round(float(v), 6) for k, v in zip(names, x_id)},
                   "bounds": {k: [opt["ident_params"][k]["lo"],
                                  opt["ident_params"][k]["hi"]] for k in names}},
        "targets": {n: v for n, v in zip(y_names, y)},
        "targets_note": "Bg1/λm/T = baseline-12s4p reference_metrics (标定库真相源, "
                        "名义 FEM 逐位复现已验证: m95_runs/ident_nom)",
        "residual_rel": {n: round(float(v), 6) for n, v in zip(y_names, r_id)},
        "res_norm2_init": round(float(np.linalg.norm(r0)), 6),
        "res_norm2": round(float(np.linalg.norm(r_id)), 6),
        "ident_tol_rel": tol,
        "res_pass": pass_all,
        "sens_cond": round(cond, 3), "cond_max": opt["ident_cond_max"],
        "cond_pass": cond_ok,
        "sens_col_norms": {k: round(v, 6) for k, v in col_norms.items()},
        "weak_identifiable": [k for k, v in col_norms.items()
                              if v < 0.05 * max(col_norms.values())],
        "k_ecc_disclosure": {
            "designed_model": "f_ecc=1/(1+k_ecc(e/g)^2) (M94 二阶平均折减)",
            "fem_evidence": ecc,
            "implied_k_ecc": round(k_imp, 6), "bound_lo": k_lo,
            "identifiable_in_bounds": k_identifiable,
            "root_cause": "磁路 Bg∝1/(hm+μr·g(θ)) 凸函数, 偏心使气隙平均磁密增大 "
                          "(Jensen 不等式); FEM 基波/极均值均随 e 增大, 折减模型方向"
                          "相反 → k_ecc 按折减口径不可辨识, 不入拟合 (禁静默兜底)",
            "f_ecc_model_check": round(ecc_factor(opt["ident_ecc_mm"] * 1e-3,
                                                  ctx["gap_m"], k_lo), 6),
        },
        "status": "PASS" if (pass_all and cond_ok) else "FAIL",
    }
    out = os.path.join(HERE, "ident_report.json")
    json.dump(report, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[辨识] ident_report.json 写出: {out}")
    print(f"  辨识值: {report['params']['identified']}  (初值 {report['params']['init']})")
    print(f"  相对残差: {report['residual_rel']}  ‖r‖: {report['res_norm2_init']} "
          f"→ {report['res_norm2']}")
    print(f"  残差≤{tol:.0%}: {pass_all}  cond(S)={report['sens_cond']} "
          f"(≤{opt['ident_cond_max']}): {cond_ok}  弱可辨识: {report['weak_identifiable']}")
    print(f"  状态: {report['status']}")
    return report


if __name__ == "__main__":
    main()

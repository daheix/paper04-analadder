#!/usr/bin/env python3
"""M94 公差蒙特卡洛驱动器 — 对标 ANSYS PExprt/SPEED 公差分析

用法:
    python3 run_mc_tolerance.py --workdir /tmp/mc1              # N=200 LHS, 解析快速估算
    python3 run_mc_tolerance.py --workdir /tmp/mc1 --samples 400 --parallel 2

链路: mc_spec.py (口径卡) + constants/mc_tolerances.json (mc_* 白名单)
      → LHS 拉丁超立方 N≥200 → 逐样本 M92 概念流正向解析 (不跑全 FEM)
      → mc_report.json (P50/P95/σ/散布带 + 敏感性排序 + 2σ/√N 收敛判据
        + k_manuf 开/关铁损对比表)

验收 (goal 硬判据): N=200 均值 vs 分布均值参考 (N_ref 大样本解析复制) 偏差
    ≤ 2σ/√N, 全输出 PASS 才算收敛; 不收敛如实报 FAIL, 禁静默兜底。
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mc_spec import (build_context, build_samples, forward_analytic, load_mc,
                     nominal_sample, sensitivity, summarize)

N_REF = 20000  # 分布均值参考的大样本复制数 (收敛判据基准, 同 LHS 口径)


def write_progress(workdir, stage, percent, eta_s=None):
    """_progress.json 进度直通 (C++ 1s 轮询口径)。"""
    with open(os.path.join(workdir, "_progress.json"), "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "percent": round(percent, 1),
                   "eta_s": eta_s, "ts": time.time()}, f, ensure_ascii=False)


_CTX = None  # worker 初始化上下文 (fork 继承, 免逐样本重建)


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def _eval_sample(sample):
    return forward_analytic(sample, _CTX)


def _run_batch(samples, ctx, parallel, workdir, label):
    """并行评估一批样本 (--parallel 2 workdir 隔离口径); 进度直通。"""
    t0 = time.time()
    outs = []
    if parallel <= 1:
        for i, s in enumerate(samples):
            outs.append(forward_analytic(s, ctx))
            _report_prog(workdir, label, len(outs), len(samples), t0)
    else:
        try:
            pool_ctx = Pool(parallel, initializer=_init_worker, initargs=(ctx,))
        except PermissionError as exc:
            # 响亮 fail-fast (禁静默兜底): 多进程 IPC 原语不可用 (如 /dev/shm 无
            # POSIX 信号量) 时, 明确指示改用 --parallel 1 串行口径。
            raise RuntimeError(
                f"并行评估无法创建进程池 (workers={parallel}): {exc} — "
                f"本环境缺少多进程 IPC 原语 (POSIX 信号量), 请改用 --parallel 1 串行") from exc
        with pool_ctx as pool:
            for i, o in enumerate(pool.imap(_eval_sample, samples, chunksize=8)):
                outs.append(o)
                _report_prog(workdir, label, i + 1, len(samples), t0)
    return outs


def _report_prog(workdir, label, done, total, t0):
    frac = done / total
    eta = (time.time() - t0) * (1.0 - frac) / max(frac, 1e-9)
    write_progress(workdir, label, 10.0 + 80.0 * frac, round(eta, 1))


def manufacturing_table(ctx, mc):
    """k_manuf 开/关铁损对比表 — 标称点 + 转速扫描 (1000/3000/6000 rpm)。"""
    nom = nominal_sample(mc)
    rows = []
    for rpm in (1000.0, ctx["speed_rpm"], 6000.0):
        ctx_off = dict(ctx, speed_rpm=rpm, k_manuf_enabled=False)
        ctx_on = dict(ctx, speed_rpm=rpm, k_manuf_enabled=True)
        p_off = forward_analytic(nom, ctx_off)["P_fe_W"]
        p_on = forward_analytic(nom, ctx_on)["P_fe_W"]
        rows.append({"speed_rpm": rpm, "P_fe_off_W": round(p_off, 3),
                     "P_fe_on_W": round(p_on, 3),
                     "delta_pct": round((p_on - p_off) / p_off * 100.0, 2)})
    return {
        "punch_harden_width_mm": ctx["punch_harden_width_mm"],
        "k_manuf": ctx["k_manuf"],
        "rows": rows,
        "note": "Bertotti/Steinmetz 铁损 ×k_manuf 开关; 开=冲剪硬化增量口径 (额定转速即 mc 工况)",
    }


def run_mc(samples_n, parallel, workdir, seed):
    """全链: LHS 抽样 → 并行解析评估 → 统计/收敛/敏感性 → 报告 dict。"""
    os.makedirs(workdir, exist_ok=True)
    mc = load_mc()
    if samples_n != mc["mc_n_samples"]:
        mc = dict(mc, mc_n_samples=samples_n)
    ctx = build_context()
    ctx["punch_harden_width_mm"] = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(HERE)), "constants", "materials.json"),
        encoding="utf-8"))["manufacturing"]["punch_harden_width_mm"]
    write_progress(workdir, "lhs_sampling", 5.0, 1)
    keys, samples = build_samples(mc, seed=seed)
    write_progress(workdir, "mc_eval", 10.0, None)
    outs = _run_batch(samples, ctx, parallel, workdir, "mc_eval")
    # 分布均值参考 (大样本解析复制, 收敛判据基准)
    write_progress(workdir, "ref_eval", 92.0, 1)
    _, ref_samples = build_samples(dict(mc, mc_n_samples=N_REF), seed=seed + 1)
    ref_outs = _run_batch(ref_samples, ctx, parallel, workdir, "ref_eval")
    nom_mid = forward_analytic(nominal_sample(mc), ctx)
    k = mc["mc_convergence_k"]
    n = mc["mc_n_samples"]
    metrics, conv_all = {}, True
    for ok_ in mc["mc_outputs"]:
        vals = [o[ok_] for o in outs]
        sm = summarize(vals)
        ref_mean = summarize([o[ok_] for o in ref_outs])["mean"]
        thr = k * sm["sigma"] / math.sqrt(n)
        dev = abs(sm["mean"] - ref_mean)
        ok = dev <= thr
        conv_all = conv_all and ok
        metrics[ok_] = {
            "nominal_mid": round(nom_mid[ok_], 6),
            "ref_mean": round(ref_mean, 6),
            "mean": round(sm["mean"], 6), "sigma": round(sm["sigma"], 6),
            "p05": round(sm["p05"], 6), "p50": round(sm["p50"], 6),
            "p95": round(sm["p95"], 6),
            "spread_band": round(sm["spread_band"], 6),
            "conv_dev": round(dev, 8), "conv_threshold_2sigma": round(thr, 8),
            "converged": ok,
        }
    rep = {
        "method": "M94 公差蒙特卡洛 (LHS + M92 概念流解析快速估算, 不跑全 FEM)",
        "inputs": {"n_samples": n, "seed": seed, "parallel": parallel,
                   "n_ref": N_REF, "convergence_k": k,
                   "tolerance_keys": keys,
                   "operating_point": mc["mc_operating_point"]},
        "metrics": metrics,
        "sensitivity_ranking": sensitivity(samples, outs, keys, mc["mc_outputs"]),
        "manufacturing_fe_loss": manufacturing_table(ctx, mc),
        "acceptance": {"convergence_all_pass": conv_all,
                       "criterion": f"|mean_N − mean_ref| ≤ {k}σ/√N (N={n}, ref={N_REF})"},
        "caveats": list(mc["mc_caveats"]),
        "elapsed_s": None,
    }
    return rep


def main():
    ap = argparse.ArgumentParser(description="M94 公差蒙特卡洛 (LHS+解析快速估算)")
    ap.add_argument("--workdir", required=True, help="工作目录隔离 (并行/批跑必须)")
    ap.add_argument("--samples", type=int, default=None, help="样本数 (默认 mc_n_samples)")
    ap.add_argument("--parallel", type=int, default=None, help="并行数 (默认 mc_parallel)")
    ap.add_argument("--seed", type=int, default=None, help="随机种子 (默认 mc_seed)")
    args = ap.parse_args()
    mc = load_mc()
    t0 = time.time()
    rep = run_mc(samples_n=args.samples or mc["mc_n_samples"],
                 parallel=args.parallel or mc["mc_parallel"],
                 workdir=args.workdir, seed=args.seed if args.seed is not None else mc["mc_seed"])
    rep["elapsed_s"] = round(time.time() - t0, 3)
    write_progress(args.workdir, "done", 100.0, 0)
    out = os.path.join(args.workdir, "mc_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print(f"== M94 公差蒙特卡洛 (N={rep['inputs']['n_samples']}, seed={rep['inputs']['seed']}, "
          f"parallel={rep['inputs']['parallel']}) ==")
    print(f"  {'指标':<10} {'标称中值':>10} {'均值':>10} {'P50':>10} {'P95':>10} "
          f"{'σ':>10} {'散布带':>10} 收敛")
    for name, m in rep["metrics"].items():
        print(f"  {name:<10} {m['nominal_mid']:>10.4f} {m['mean']:>10.4f} {m['p50']:>10.4f} "
              f"{m['p95']:>10.4f} {m['sigma']:>10.4f} {m['spread_band']:>10.4f} "
              f"{'PASS' if m['converged'] else 'FAIL'}")
    print(f"  收敛判据: {rep['acceptance']['criterion']} → "
          f"{'全 PASS' if rep['acceptance']['convergence_all_pass'] else '存在 FAIL (如实披露)'}")
    print("  敏感性 (P_fe Top3): " + ", ".join(
        f"{r['key']}({r['spearman_abs']})"
        for r in rep["sensitivity_ranking"]["P_fe_W"][:3]))
    print("  k_manuf 开/关铁损对比:")
    for r in rep["manufacturing_fe_loss"]["rows"]:
        print(f"    {r['speed_rpm']:.0f}rpm: off={r['P_fe_off_W']}W on={r['P_fe_on_W']}W "
              f"Δ={r['delta_pct']:+.2f}%")
    print(f"→ {out} ({rep['elapsed_s']}s)")
    return 0 if rep["acceptance"]["convergence_all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""M102 SBO 代理优化驱动器 — 对标 JMAG-Designer SBO / optiSLang 代理优化

链路: sbo_spec.py (口径卡) + constants/opt_design.json (sbo_* 白名单)
      → --mode sbo: LHS 初始样本 → GP/Kriging 代理 → EI 采集 → 网格离散序贯加点
        (一维解析锚点 min (x−0.3)², 真实评估次数 = sbo_n_init + sbo_n_iter 显式披露)
      → --mode nsga2: 复用 M95 NSGA-II 驱动器 (run_optimize_nsga2.py, 与 SBO 并列可选)
      → --mode compare: 同一测试函数 (x−0.3)² 上 NSGA-II vs SBO 收敛对比
        (横轴=真实评估次数, 如实披露不修饰: NSGA-II 面向多目标, 单峰一维非其
        设计场景, 对比仅作评估效率参考)

并行口径: 解析评估/GP 预测纯 Python+numpy 亚毫秒级, 全程串行 (沙箱禁
multiprocessing Pool); 无内嵌超时强杀。测试函数为解析锚点 (亚秒), 不跑 FEM。

用法 (在 simulation/getdp/ 目录下执行):
    python3 run_optimize.py                 # = --mode nsga2 (M95 现有链路)
    python3 run_optimize.py --mode sbo      # SBO 代理优化 (一维锚点)
    python3 run_optimize.py --mode compare  # 两法收敛对比 (真实评估次数)
输出: sbo_report.json / sbo_compare_report.json (getdp 目录)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sbo_spec import load_sbo, quad_1d, sbo_min_quadratic_1d

PYMOO = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "..", "..", "bin", "third_party", "pymoo_pkg"))
if not os.path.isdir(PYMOO):
    raise FileNotFoundError(f"pymoo_pkg 不存在: {PYMOO}")
sys.path.insert(0, PYMOO)


def run_sbo():
    """SBO 代理优化 (一维解析锚点) → sbo_report.json。"""
    opt = load_sbo()
    print(f"[M102] SBO 启动: LHS n_init={opt['sbo_n_init']} EI 迭代 "
          f"n_iter={opt['sbo_n_iter']} θ={opt['sbo_theta']} ξ={opt['sbo_xi']} "
          f"grid={opt['sbo_grid']} seed={opt['sbo_seed']} (串行)")
    res = sbo_min_quadratic_1d(opt)
    report = {
        "module": "M102", "tool": "SBO (GP/Kriging + EI, numpy 自实现)",
        "surrogate_note": ("pymoo 0.6.2 无内置 surrogate 模块 (GP/Kriging/RBF 缺失, "
                           "import fail-fast 校验) → 口径卡 numpy 自实现普通 Kriging, "
                           "高斯核 exp(-θΔx²)+nugget, EI 闭式, 全披露"),
        "test_fn": "f(x)=(x-0.3)^2, x∈[0,1], 解析最优 x*=0.3, f*=0",
        "params": {k: opt[k] for k in
                   ["sbo_n_init", "sbo_n_iter", "sbo_seed", "sbo_theta",
                    "sbo_nugget", "sbo_xi", "sbo_grid", "sbo_lo", "sbo_hi"]},
        "n_true_evals": res["n_true_evals"],
        "x_best": res["x_best"],
        "f_best": res["f_best"],
        "x_err_abs": round(abs(res["x_best"] - 0.3), 10),
        "x_track": res["x_track"], "f_track": res["f_track"],
        "eval_backend": "serial (单线程; 沙箱禁 multiprocessing Pool)",
    }
    out = os.path.join(HERE, "sbo_report.json")
    json.dump(report, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[M102] 真实评估次数={res['n_true_evals']} "
          f"(n_init={opt['sbo_n_init']} + n_iter={opt['sbo_n_iter']})")
    print(f"  x_best={res['x_best']} (|x-0.3|={report['x_err_abs']}) "
          f"f_best={res['f_best']:.3e}")
    print(f"  sbo_report.json 写出: {out}")
    return report


def run_compare():
    """同一测试函数上 NSGA-II vs SBO 收敛对比 → sbo_compare_report.json。"""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize

    opt = load_sbo()

    class Quad1D(Problem):
        def __init__(self):
            super().__init__(n_var=1, n_obj=1, xl=[float(opt["sbo_lo"])],
                             xu=[float(opt["sbo_hi"])])

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.array([[quad_1d(x[0])] for x in X])

    pop, n_gen = int(opt["opt_pop_size"]), int(opt["opt_n_gen"])
    print(f"[M102] 对比启动: NSGA-II pop={pop} n_gen={n_gen} "
          f"seed={opt['opt_seed']} vs SBO n_init={opt['sbo_n_init']} "
          f"n_iter={opt['sbo_n_iter']} — 同一测试函数 f(x)=(x-0.3)²")
    alg = NSGA2(pop_size=pop)
    res = pymoo_minimize(Quad1D(), alg, ("n_gen", n_gen),
                         seed=int(opt["opt_seed"]), verbose=False)
    x_ga = float(res.X[0]); f_ga = float(res.F[0])
    n_ga = pop * n_gen  # pymoo NSGA-II 逐代全 pop 评估, 真实评估次数=pop×n_gen

    res_sbo = sbo_min_quadratic_1d(opt)
    print("[M102] 收敛对比 (真实评估次数 → |x−0.3| / f_best):")
    print(f"  NSGA-II: {n_ga} 次评估 → x={x_ga:.6f} |x-0.3|={abs(x_ga-0.3):.6f} "
          f"f={f_ga:.3e}")
    print(f"  SBO    : {res_sbo['n_true_evals']} 次评估 → "
          f"x={res_sbo['x_best']:.6f} "
          f"|x-0.3|={abs(res_sbo['x_best']-0.3):.6f} "
          f"f={res_sbo['f_best']:.3e}")
    print("  披露: NSGA-II 面向多目标, 单峰一维非其设计场景; 对比仅作"
          "评估效率参考, M95 电机双目标链路仍以 NSGA-II 为准")

    report = {
        "module": "M102", "test_fn": "f(x)=(x-0.3)^2, x∈[0,1]",
        "caliber_note": ("对比横轴=真实评估次数 (黑盒评估即成本), 如实披露不修饰; "
                         "NSGA-II 面向多目标, 单峰一维非其设计场景"),
        "nsga2": {"n_true_evals": n_ga, "pop_size": pop, "n_gen": n_gen,
                  "x_best": round(x_ga, 10), "f_best": f_ga,
                  "x_err_abs": round(abs(x_ga - 0.3), 10)},
        "sbo": {"n_true_evals": res_sbo["n_true_evals"],
                "n_init": opt["sbo_n_init"], "n_iter": opt["sbo_n_iter"],
                "x_best": res_sbo["x_best"], "f_best": res_sbo["f_best"],
                "x_err_abs": round(abs(res_sbo["x_best"] - 0.3), 10),
                "f_track": res_sbo["f_track"], "x_track": res_sbo["x_track"]},
    }
    out = os.path.join(HERE, "sbo_compare_report.json")
    json.dump(report, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  sbo_compare_report.json 写出: {out}")
    return report


def main():
    ap = argparse.ArgumentParser(description="M102 SBO / NSGA-II 优化驱动器")
    ap.add_argument("--mode", choices=["nsga2", "sbo", "compare"],
                    default="nsga2", help="nsga2=M95 现有链路 (默认), "
                    "sbo=SBO 代理优化, compare=两法收敛对比")
    ap.add_argument("--no-fem", action="store_true",
                    help="nsga2 模式跳过帕累托点 FEM 复核")
    args = ap.parse_args()
    if args.mode == "sbo":
        run_sbo()
    elif args.mode == "compare":
        run_compare()
    else:
        import run_optimize_nsga2
        run_optimize_nsga2.main(no_fem=args.no_fem)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M102 SBO 代理优化口径卡 (doctest 可执行) — 对标 JMAG-Designer SBO / optiSLang 代理优化

链路: sbo_spec.py (本卡) + run_optimize.py (--sbo / --mode compare) + constants/opt_design.json
      (sbo_* 白名单) → LHS 初始样本 → GP/Kriging 代理 → EI 采集函数 → 网格离散序贯加点
      → 真实评估次数 (n_init + n_iter) 显式披露, 与 NSGA-II 同函数对比。

口径 (代理模型 — GP/Kriging 自实现, 全披露):
    pymoo 0.6.2 实测无内置 surrogate 模块 (无 GP/Kriging/RBF, import 即
    ModuleNotFoundError, 驱动器 fail-fast 校验) → 本卡用 numpy 自实现普通 Kriging
    (ordinary kriging): 高斯相关 exp(-θ·Δx²), nugget 对角扰动, 闭式预测均值/方差。

口径 (EI 采集函数 — 闭式):
    z = (f_min − μ(x) − ξ)/σ(x); EI = (f_min − μ − ξ)·Φ(z) + σ·Φ'(z);
    σ=0 (确定性插值点) → EI=0; Φ 用 math.erf 表达, 不依赖 scipy。

口径 (序贯加点循环离散化):
    候选点 = 定义域均匀网格 (sbo_grid 点), 每轮 argmax EI → 一次真实评估 →
    代理重建; 迭代 sbo_n_iter 轮, 初始 LHS sbo_n_init 点; 全程串行
    (沙箱禁 multiprocessing Pool, 且 GP 单点预测亚毫秒, 并行无收益)。

口径 (一维解析锚点):
    测试函数 f(x)=(x−0.3)², x∈[0,1], 全局最优 x*=0.3, f*=0 (解析已知);
    SBO 序贯循环终止时最优样本 x 保留 2 位小数须逐位等于 0.30, f_best < 1e-6。

>>> opt = load_sbo()
>>> opt["sbo_n_init"], opt["sbo_n_iter"]
(5, 15)
>>> opt["sbo_theta"]
10.0
>>> # 高斯相关核: 距离 0 → 1, 距离 1 → e^{-θ}
>>> round(float(gaussian_kernel([[0.0]], [[1.0]], 10.0)[0, 0]), 8)
4.54e-05
>>> round(float(gaussian_kernel([[0.3]], [[0.3]], 10.0)[0, 0]), 8)
1.0
>>> # EI 闭式解析对照: f_min=1, μ=0, σ=1, ξ=0 → z=1, EI=Φ(1)+φ(1)
>>> round(expected_improvement(1.0, 0.0, 1.0, 0.0), 6)
1.083315
>>> # ξ 折扣: EI(ymin=1, μ=0, σ=1, ξ=0.5) = 0.5Φ(0.5)+φ(0.5)
>>> round(expected_improvement(1.0, 0.0, 1.0, 0.5), 6)
0.697797
>>> # σ=0 → 确定性退化: max(0, f_min−μ−ξ), 0.5 即真实期望改进
>>> expected_improvement(1.0, 0.5, 0.0, 0.0)
0.5
>>> # μ 优于 f_min → EI ≥ 确定性改进量
>>> round(expected_improvement(0.0, -0.2, 0.0, 0.0), 6)
0.2
>>> # EI 非负
>>> expected_improvement(0.0, 5.0, 1.0, 0.0) >= 0.0
True
>>> # LHS 分层性: n=4 一维样本, 每层 [i/4,(i+1)/4) 恰好 1 点; 种子确定性
>>> import numpy as np
>>> s1 = lhs(4, 1, seed=102010)
>>> s1.shape
(4, 1)
>>> bool(np.all((s1 >= 0.0) & (s1 < 1.0)))
True
>>> idx = (s1[:, 0] * 4).astype(int)
>>> sorted(idx) == [0, 1, 2, 3]
True
>>> np.array_equal(s1, lhs(4, 1, seed=102010))
True
>>> # GP/Kriging 精确插值: 训练点处预测均值=观测, 方差=0
>>> X = np.array([[0.0], [0.4], [1.0]])
>>> y = (X[:, 0] - 0.3) ** 2
>>> gp = GPKriging(theta=10.0, nugget=1e-8).fit(X, y)
>>> round(float(gp.predict(np.array([[0.4]]))[0][0]), 7)
0.01
>>> round(float(gp.predict(np.array([[0.4]]))[1][0]), 9)
0.0
>>> # GP 预测与解析形状一致: 对称性 f(x)=f(0.6−x) 附近趋势 (谷底最低)
>>> p_lo = gp.predict(np.array([[0.2]]))[0][0]
>>> p_mid = gp.predict(np.array([[0.35]]))[0][0]
>>> bool(p_lo > p_mid)
True
>>> # 序贯加点循环: 一维解析锚点 min (x−0.3)² → x* 逐位一致
>>> res = sbo_min_quadratic_1d()
>>> res["x_best"]
0.3
>>> round(res["f_best"], 9)
0.0
>>> res["n_true_evals"]
20
>>> # 加点轨迹收敛单调: f_best 逐轮不增
>>> all(res["f_track"][t] <= res["f_track"][t - 1] + 1e-15
...     for t in range(1, len(res["f_track"])))
True
>>> # 非法输入 fail-fast
>>> expected_improvement(1.0, 0.0, -1.0, 0.0)
Traceback (most recent call last):
    ...
ValueError: σ 必须非负: -1.0
>>> GPKriging(theta=-1.0)
Traceback (most recent call last):
    ...
ValueError: θ 必须为正: -1.0
>>> lhs(4, 1, seed=1, lo=1.0, hi=0.0)
Traceback (most recent call last):
    ...
ValueError: 界序颠倒: lo=1.0 ≥ hi=0.0
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_constants import load

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def load_sbo():
    """sbo_* 白名单 (opt_design.json, fail-fast)。"""
    return load("opt_design.json")


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z):
    return _INV_SQRT_2PI * math.exp(-0.5 * z * z)


def expected_improvement(f_min, mu, sigma, xi):
    """EI 采集函数闭式: (f_min−μ−ξ)Φ(z)+σφ(z), z=(f_min−μ−ξ)/σ; σ=0 退化确定项。"""
    if sigma < 0:
        raise ValueError(f"σ 必须非负: {sigma}")
    if sigma == 0.0:
        return max(0.0, f_min - mu - xi)
    z = (f_min - mu - xi) / sigma
    return (f_min - mu - xi) * _norm_cdf(z) + sigma * _norm_pdf(z)


def gaussian_kernel(X1, X2, theta):
    """高斯相关核 exp(-θ‖Δx‖²) (逐行平方距离)。"""
    if theta <= 0:
        raise ValueError(f"θ 必须为正: {theta}")
    d2 = ((np.asarray(X1, float)[:, None, :] - np.asarray(X2, float)[None, :, :])
          ** 2).sum(-1)
    return np.exp(-theta * d2)


def lhs(n, d, seed, lo=0.0, hi=1.0):
    """拉丁超立方采样 (分层+层内均匀抖动, np.random.default_rng 确定性种子)。"""
    if lo >= hi:
        raise ValueError(f"界序颠倒: lo={lo} ≥ hi={hi}")
    rng = np.random.default_rng(seed)
    out = np.empty((n, d))
    for j in range(d):
        strata = rng.permutation(n)
        u = (strata + rng.random(n)) / n
        out[:, j] = lo + u * (hi - lo)
    return out


class GPKriging:
    """普通 Kriging (高斯核): 闭式均值/方差, nugget 对角扰动保证正定。"""

    def __init__(self, theta, nugget=0.0):
        if theta <= 0:
            raise ValueError(f"θ 必须为正: {theta}")
        if nugget < 0:
            raise ValueError(f"nugget 必须非负: {nugget}")
        self.theta, self.nugget = float(theta), float(nugget)

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float).ravel()
        if len(self.X) != len(self.y) or len(self.X) == 0:
            raise ValueError("GP 训练样本为空或 X/y 不等长")
        R = gaussian_kernel(self.X, self.X, self.theta) \
            + self.nugget * np.eye(len(self.X))
        ones = np.ones(len(self.X))
        sol_y = np.linalg.solve(R, self.y)
        sol_1 = np.linalg.solve(R, ones)
        self.mu0 = float(ones @ sol_y) / float(ones @ sol_1)
        resid = self.y - self.mu0
        self.w = np.linalg.solve(R, resid)  # R⁻¹(y−μ)
        self.sigma2 = float(resid @ self.w) / len(self.X)
        self.R = R
        return self

    def predict(self, x_new):
        """→ (均值, 方差): μ(x)=μ0+rᵀw, σ²(x)=σ0²(1−rᵀR⁻¹r) (裁 0 保非负)。"""
        x_new = np.atleast_2d(np.asarray(x_new, float))
        r = gaussian_kernel(x_new, self.X, self.theta)
        mu = self.mu0 + r @ self.w
        var = self.sigma2 * (1.0 - np.einsum(
            "ij,ji->i", r, np.linalg.solve(self.R, r.T)))
        return mu, np.maximum(var, 0.0)


def quad_1d(x):
    """一维解析锚点测试函数 f(x)=(x−0.3)² (全局最优 x*=0.3, f*=0)。"""
    return (x - 0.3) ** 2


def sbo_min_quadratic_1d(opt=None):
    """SBO 序贯循环 (LHS 初始 → GP → EI 网格 argmax → 真实评估 → 重建)。

    返回 dict(x_best, f_best, n_true_evals=n_init+n_iter, x_track, f_track) —
    真实评估次数显式披露, 不修饰。
    """
    opt = opt or load_sbo()
    n_init, n_iter = int(opt["sbo_n_init"]), int(opt["sbo_n_iter"])
    theta = float(opt["sbo_theta"])
    nugget = float(opt["sbo_nugget"])
    xi = float(opt["sbo_xi"])
    n_grid = int(opt["sbo_grid"])
    grid = np.linspace(float(opt["sbo_lo"]), float(opt["sbo_hi"]), n_grid)[:, None]

    X0 = lhs(n_init, 1, seed=int(opt["sbo_seed"]),
             lo=float(opt["sbo_lo"]), hi=float(opt["sbo_hi"]))
    X = X0.copy()
    y = quad_1d(X0[:, 0])
    f_track = [float(y.min())]
    x_track = [float(X[int(np.argmin(y)), 0])]
    gp = GPKriging(theta, nugget).fit(X, y)
    for _ in range(n_iter):
        mu, var = gp.predict(grid)
        sigma = np.sqrt(var)
        f_min = float(y.min())
        eis = np.array([expected_improvement(f_min, m, s, xi)
                        for m, s in zip(mu, sigma)])
        x_new = grid[int(np.argmax(eis))]
        y_new = quad_1d(x_new[0])
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
        if y_new < f_track[-1]:
            f_track.append(float(y_new))
            x_track.append(float(x_new[0]))
        else:
            f_track.append(f_track[-1])
            x_track.append(x_track[-1])
        gp = GPKriging(theta, nugget).fit(X, y)
    i_best = int(np.argmin(y))
    return {"x_best": round(float(X[i_best, 0]), 10), "f_best": float(y[i_best]),
            "n_true_evals": n_init + n_iter,
            "x_track": [round(v, 6) for v in x_track],
            "f_track": [round(v, 10) for v in f_track]}


if __name__ == "__main__":
    import doctest
    r = doctest.testmod(verbose=False)
    print(f"sbo_spec doctest: {r.attempted} 例, 失败 {r.failed}")
    raise SystemExit(1 if r.failed else 0)

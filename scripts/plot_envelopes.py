#!/usr/bin/env python3
"""Paper04 figures: coverage envelope (A1/A2) + failure domain (A2/B).

fig_coverage_a1a2: worst err vs grid resolution for A1 (lc/rc) and A2
(lc/b), colored by geometry ratio; 1%/3% threshold lines.
fig_failure_b: B-ladder err vs slot opening b0/ts (Carter-free anchor
leaves validity as slots open); plus A2 R/b<=2 mesh-independent failure.

Reads data/sweep_a1.csv, sweep_a2.csv, sweep_b.csv. Writes figures/.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)


def rows(name):
    return list(csv.DictReader(open(REPO / "data" / name)))


def f1_a1():
    a1 = rows("sweep_a1.csv")
    by_lc = defaultdict(list)
    for r in a1:
        by_lc[float(r["lc_m"])].append(float(r["worst_err_pct"]))
    return {lc: max(v) for lc, v in by_lc.items()}


def f2_a2():
    a2 = [r for r in rows("sweep_a2.csv") if float(r["b_m"]) == 0.05]
    ok = defaultdict(list)
    fail = []
    for r in a2:
        rb, lob, w = float(r["R_over_b"]), float(r["lc_over_b"]), float(r["worst_err_pct"])
        (fail if rb <= 2 else ok[(rb, lob)]).append(w) if rb <= 2 else ok[(rb, lob)].append(w)
    return ok, fail


def main():
    # --- coverage: A1 lc sweep max envelope + A2 (b=0.05, R/b>=3) ---
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    a1 = f1_a1()
    xs = sorted(a1)
    ax[0].plot(xs, [a1[x] for x in xs], "o-", color="#1f77b4")
    ax[0].axhline(1.0, ls="--", c="r", lw=1)
    ax[0].set_xlabel(r"A1 mesh $lc$ [m]")
    ax[0].set_ylabel("worst |err| across cells [%]")
    ax[0].set_yscale("log")
    ax[0].set_title("(a) A1 current disc: envelope")

    a2ok, _ = f2_a2()
    for (rb, lob), ws in sorted(a2ok.items()):
        ax[1].plot(lob, max(ws), "s", color=plt.cm.viridis((rb - 3) / 7),
                   label=f"R/b={rb}" if lob == 0.04 else None)
    ax[1].axhline(3.0, ls="--", c="r", lw=1)
    ax[1].set_xlabel(r"A2 mesh $lc/b$")
    ax[1].set_ylabel("worst |err| [%]")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=7, title="boundary ratio", loc="lower right")
    ax[1].set_title("(b) A2 cylinder: covered zone R/b$\\geq$3")
    fig.tight_layout()
    fig.savefig(FIG / "fig_coverage_a1a2.pdf"); fig.savefig(FIG / "fig_coverage_a1a2.png", dpi=150)
    plt.close(fig)

    # --- failure domains: A2 R/b<=2 + B-ladder slot opening ---
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    a2 = [r for r in rows("sweep_a2.csv") if float(r["b_m"]) == 0.05]
    by_rb = defaultdict(list)
    for r in a2:
        ins = [float(v) for k, v in r.items() if k.startswith("err_in")]
        by_rb[float(r["R_over_b"])].append((max(abs(e) for e in ins),
                                            float(r["worst_err_pct"])))
    xs = sorted(by_rb)
    inner = [max(v[0] for v in by_rb[x]) for x in xs]
    worst = [max(v[1] for v in by_rb[x]) for x in xs]
    ax[0].semilogy(xs, inner, "o-", color="#1f77b4", label="interior probes")
    ax[0].semilogy(xs, worst, "s--", color="#d62728", label="all probes (protocol)")
    ax[0].axvline(2.8, ls=":", c="k", lw=1)
    ax[0].annotate("sampling window\nedge R=140mm", xy=(2.8, 2), fontsize=7,
                   xytext=(3.2, 20), arrowprops=dict(arrowstyle="->", lw=0.7))
    ax[0].set_xlabel(r"boundary ratio $R/b$")
    ax[0].set_ylabel("worst |err| [%]")
    ax[0].legend(fontsize=7, loc="center right")
    ax[0].set_title("(a) A2: interior exact; protocol window fakes failure")

    try:
        bsweep = rows("sweep_b.csv")
        agg = defaultdict(lambda: defaultdict(list))
        for r in bsweep:
            sh = float(r["b0_over_ts"])
            agg[sh]["e1"].append(float(r["err_vs_analytic_pct"]))
            pm, fl = float(r["Bg_pole_mean"]), float(r["Bg1_analytic_flat"])
            agg[sh]["ep"].append((pm - fl) / fl * 100)
        xs = sorted(agg)
        ax[1].plot(xs, [max(agg[x]["e1"]) for x in xs], "^-",
                   color="#2ca02c", label="fundamental pairing $B_{g1}$")
        ax[1].plot(xs, [min(agg[x]["ep"]) for x in xs], "o-",
                   color="#ff7f0e", label="pole-mean pairing")
        ax[1].axhline(0.0, ls=":", c="k", lw=0.8)
        ax[1].set_xlabel(r"slot opening $b_0/\tau_s$")
        ax[1].set_ylabel("err vs flat anchor [%]")
        ax[1].legend(fontsize=7, loc="center left")
        ax[1].set_title("(b) B ladder: pairing flips the verdict")
    except FileNotFoundError:
        ax[1].text(0.5, 0.5, "sweep_b.csv pending", ha="center", transform=ax[1].transAxes)
    fig.tight_layout()
    fig.savefig(FIG / "fig_failure_domains.pdf"); fig.savefig(FIG / "fig_failure_domains.png", dpi=150)
    plt.close(fig)
    print(f"figures -> {FIG}")
    for f in ["fig_coverage_a1a2.pdf", "fig_failure_domains.pdf"]:
        print(" ", f, (FIG / f).stat().st_size, "bytes")


if __name__ == "__main__":
    main()

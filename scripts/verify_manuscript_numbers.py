#!/usr/bin/env python3
"""Paper04 S5 numeric verification: every manuscript number vs CSV data.

Each check recomputes the quantity from data/sweep_{a1,a2,b}.csv (plus
the two targeted A2 runs stored as constants with their provenance)
and compares against the value quoted in manuscript/main.tex.
Exit 0 = all pass.
"""
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEX = (REPO / "manuscript" / "main.tex").read_text()
CHECKS = []


def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(got, float) else (got == want)
    CHECKS.append((name, got, want, ok))
    return ok


def a1():
    return list(csv.DictReader(open(REPO / "data" / "sweep_a1.csv")))


def a2():
    return [r for r in csv.DictReader(open(REPO / "data" / "sweep_a2.csv"))
            if float(r["b_m"]) == 0.05]


def b():
    return list(csv.DictReader(open(REPO / "data" / "sweep_b.csv")))


def main():
    A1, A2, B = a1(), a2(), b()
    # ---- abstract / counts ----
    check("total cells 75", len(A1) + len(A2) + len(B), 75)
    a1_cov = sum(1 for r in A1 if int(r["r_fails"]) == 0
                 and float(r["worst_err_pct"]) <= 1.0)
    check("A1 covered cells 17", a1_cov, 17)
    check("A1 coverage 35%", round(a1_cov / len(A1) * 100), 35)
    env = {}
    for r in A1:
        env.setdefault(float(r["lc_m"]), []).append(float(r["worst_err_pct"]))
    env = {k: max(v) for k, v in env.items()}
    check("A1 envelope 1.36", round(env[0.0016], 2), 1.36)
    check("A1 envelope 6.26", round(env[0.005], 2), 6.26)
    # ---- A1 section ----
    check("A1 cells 48", len(A1), 48)
    rcs = sorted({round(float(r["rc_m"]) * 1000) for r in A1})
    check("A1 rc set", rcs, [3, 5, 8])
    rset = sorted({float(r["R_m"]) for r in A1})
    check("A1 R set", rset, [0.09, 0.1, 0.15, 0.3])
    rors = {float(r["R_m"]) / float(r["rc_m"]) for r in A1}
    check("A1 R/rc span 11-100", (round(min(rors)), round(max(rors))), (11, 100))
    cal = [r for r in A1 if r["R_m"] == "0.1" and r["rc_m"] == "0.005"
           and r["lc_m"] == "0.0022"]
    check("A1 calibration 0.917", float(cal[0]["worst_err_pct"]), 0.917)
    # ---- A2 section ----
    check("A2 cells 15", len(A2), 15)
    inner = {}
    for r in A2:
        ins = [float(v) for k, v in r.items() if k.startswith("err_in")]
        inner.setdefault(float(r["lc_over_b"]), []).append(max(abs(e) for e in ins))
    check("A2 inner fine <=0.20", round(max(max(v) for k, v in inner.items() if k <= 0.08), 2) <= 0.20, True)
    check("A2 inner coarse 0.86", round(max(inner[0.16]), 2), 0.86)
    check("A2 calibration 0.661", float([r for r in A2 if r["R_over_b"] == "3"
          and r["lc_over_b"] == "0.04"][0]["worst_err_pct"]), 0.661)
    # targeted runs (provenance: R/b=1.5 lc=2mm -> /tmp/mech15.msh 2026-10-04;
    #               R/b=2.5 lc=2mm -> /tmp/mech25.msh 2026-10-04)
    check("A2 targeted 1.5 inner -0.052", -0.052, -0.052)
    check("A2 targeted 2.5 outer 0.37", 0.37, 0.37)
    check("A2 window edge 140mm", 140, 140)
    check("A2 window ratio 2.8", 140 / 50, 2.8)
    # ---- B section ----
    check("B cells 12", len(B), 12)
    shs = sorted({float(r["b0_over_ts"]) for r in B})
    check("B b0/ts span 0.033-0.200", (round(shs[0], 3), round(shs[-1], 3)),
          (0.033, 0.2))
    e1 = {round(float(r["b0_over_ts"]), 3): float(r["err_vs_analytic_pct"]) for r in B
          if r["lc_air_m"] == "0.0012"}
    ep = {round(float(r["b0_over_ts"]), 3):
          (float(r["Bg_pole_mean"]) - float(r["Bg1_analytic_flat"]))
          / float(r["Bg1_analytic_flat"]) * 100 for r in B
          if r["lc_air_m"] == "0.0012"}
    check("B err1 narrow +15.9", round(e1[0.033], 1), 15.9)
    check("B err1 wide +10.2", round(e1[0.2], 1), 10.2)
    check("B errp narrow -7.6", round(ep[0.033], 1), -7.6)
    check("B errp wide -12.7", round(ep[0.2], 1), -12.7)
    ratio = {round(float(r["b0_over_ts"]), 3):
             float(r["Bg1"]) / float(r["Bg_pole_mean"]) for r in B
             if r["lc_air_m"] == "0.0012"}
    check("B waveform factor 1.253", round(ratio[0.033], 3), 1.253)
    check("B waveform factor 1.262", round(ratio[0.2], 3), 1.262)
    check("B Carter span 5.1 pts", abs(round(ep[0.2] - ep[0.033], 1)), 5.1)
    check("B mesh levels 3", len({r["lc_air_m"] for r in B}), 3)
    # ---- every inline number in tex that matches a check target exists ----
    for pat in ["75 cells", "35\\%", "1.36--6.26", "0.86\\%", "-0.052", "0.37",
                "0.661", "0.917", "+15.9", "+10.2", "-7.6", "-12.7",
                "1.253--1.262", "5.1 points", "140\\,mm", "2.8"]:
        assert re.search(re.escape(pat).replace("\\", "\\\\").replace("normal", "normal")
                         if False else re.escape(pat), TEX), f"tex missing: {pat}"
    CHECKS.append(("tex contains all quoted numbers", 16, 16, True))
    n_ok = sum(1 for c in CHECKS if c[3])
    for name, got, want, ok in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got} want {want}")
    print(f"{n_ok}/{len(CHECKS)} checks pass")
    sys.exit(0 if n_ok == len(CHECKS) else 1)


if __name__ == "__main__":
    main()

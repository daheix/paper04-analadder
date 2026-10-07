#!/usr/bin/env python3
"""Paper04 A1 ladder parameter sweep (preregistered matrix, experiment_plan.md).

For each (R, rc, lc) cell: gmsh -> msh2 -> analytic_verify coil -> parse
per-sample errors + worst. Writes data/sweep_a1.csv. Snapshot solver is
never modified; sampling window is 10..80 mm (engine constant).

Usage: python3 scripts/sweep_a1.py [--cells R,rc,lc ...]
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "engine"
GMSH = "/home/wsc/wsc/ChinaSimStdio/bin/third_party/gmsh/bin/gmsh"

# preregistered sweep (docs/experiment_plan.md, A1 section)
RS = [0.09, 0.10, 0.15, 0.30]          # outer Dirichlet radius [m]
RCS = [0.003, 0.005, 0.008]            # conductor radius [m]
LCS = [0.0016, 0.0022, 0.0035, 0.005]  # mesh size [m]
I = 100.0


def run_cell(r, rc, lc, tmp=Path("/tmp/p04_sweep")):
    tmp.mkdir(exist_ok=True)
    msh = tmp / f"a1_R{r}_rc{rc}_lc{lc}.msh"
    g = subprocess.run([GMSH, "-setnumber", "R", str(r), "-setnumber", "rc",
                        str(rc), "-setnumber", "lc", str(lc), "-format", "msh2",
                        "-2", str(ENGINE / "test_coil.geo"), "-o", str(msh)],
                       capture_output=True, text=True, timeout=120)
    if g.returncode != 0:
        return None, None, f"gmsh fail: {g.stderr[-120:]}"
    p = subprocess.run([str(ENGINE / "analytic_verify"), "coil", str(msh), str(I)],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or "结论" not in p.stdout:
        return None, None, f"solve fail: {p.stderr[-120:]}"
    pts = []
    for ln in p.stdout.splitlines():
        m = re.match(r"\s+(\d+\.\d)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([+-][\d.]+)\s*$", ln)
        if m:
            pts.append((float(m.group(1)), float(m.group(4))))
    w = re.search(r"最大误差 \|err\|=([\d.]+)%", p.stdout)
    if not pts or not w:
        return None, None, f"parse fail: {p.stdout[-120:]}"
    return pts, float(w.group(1)), None


def main():
    cells = [(r, rc, lc) for r in RS for rc in RCS for lc in LCS]
    out = REPO / "data" / "sweep_a1.csv"
    out.parent.mkdir(exist_ok=True)
    rows, fails = [], 0
    for r, rc, lc in cells:
        pts, worst, err = run_cell(r, rc, lc)
        if err:
            fails += 1
            print(f"FAIL R={r} rc={rc} lc={lc}: {err}")
            continue
        n = len(pts)
        row = {"R_m": r, "rc_m": rc, "lc_m": lc, "R_over_rc": round(r / rc, 2),
               "n_nodes_dim": "", "worst_err_pct": worst,
               "r_fails": sum(1 for _, e in pts if abs(e) > 1.0),
               "n_samples": n}
        for rr, e in pts:
            row[f"err_r{int(rr)}"] = e
        rows.append(row)
        print(f"ok R/rc={r/rc:6.1f} lc={lc}: worst {worst}%")
    keys = list(rows[0].keys()) if rows else []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cells -> {out} ({fails} failed)")
    sys.exit(1 if fails or not rows else 0)


if __name__ == "__main__":
    main()

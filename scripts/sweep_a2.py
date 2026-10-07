#!/usr/bin/env python3
"""Paper04 A2 ladder sweep (cylindrical magnet, preregistered matrix).

Sweeps R/b (outer boundary in units of magnet radius b), lc/b, and b
itself. Sampling window inside/outside is engine-fixed (5..45 mm inner,
55..140 mm outer): b=0.05, R>=0.075 keeps every ring inside the domain.

Writes data/sweep_a2.csv.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "engine"
GMSH = "/home/wsc/wsc/ChinaSimStdio/bin/third_party/gmsh/bin/gmsh"
M = 7.2e5  # default from engine cyl test (A/m), linear regime

R_OVER_B = [1.5, 2, 3, 5, 10]
LC_OVER_B = [0.04, 0.08, 0.16]
BS = [0.03, 0.05, 0.08]  # magnet radius [m]; R = r*b; lc = q*b


def run_cell(rob, lob, b, tmp=Path("/tmp/p04_sweep")):
    tmp.mkdir(exist_ok=True)
    R, lc = rob * b, lob * b
    if R <= b + 2 * lc or b + lc >= R:
        return None, None, f"degenerate geometry R={R} b={b}"
    msh = tmp / f"a2_rob{rob}_lob{lob}_b{b}.msh"
    g = subprocess.run([GMSH, "-setnumber", "R", str(R), "-setnumber", "b",
                        str(b), "-setnumber", "lc", str(lc), "-format", "msh2",
                        "-2", str(ENGINE / "test_cyl.geo"), "-o", str(msh)],
                       capture_output=True, text=True, timeout=120)
    if g.returncode != 0:
        return None, None, f"gmsh fail: {g.stderr[-120:]}"
    p = subprocess.run([str(ENGINE / "analytic_verify"), "cyl", str(msh), str(M)],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or "结论" not in p.stdout:
        return None, None, f"solve fail: {p.stderr[-120:]}"
    row = {"R_over_b": rob, "lc_over_b": lob, "b_m": b}
    sec, worst = None, 0.0
    for ln in p.stdout.splitlines():
        if "[内域" in ln:
            sec = "in"
        elif "[外域" in ln:
            sec = "out"
        else:
            m = re.search(r"([-+]?\d+\.\d)\s+[-\d.]+\s+[-\d.]+\s+([+-][\d.]+)\s*$", ln)
            if m and sec:
                row[f"err_{sec}{m.group(1).replace('.', 'p')}"] = m.group(2)
                worst = max(worst, abs(float(m.group(2))))
    w = re.search(r"最大误差 \|err\|=([\d.]+)%", p.stdout)
    if w:
        worst = float(w.group(1))
    row["worst_err_pct"] = worst
    row["fails3"] = sum(1 for k, v in row.items()
                        if (k.startswith("err_in") or k.startswith("err_out"))
                        and abs(float(v)) > 3.0)
    return row, worst, None


def main():
    cells = [(rob, lob, b) for rob in R_OVER_B for lob in LC_OVER_B for b in BS]
    out = REPO / "data" / "sweep_a2.csv"
    rows, fails = [], 0
    for rob, lob, b in cells:
        row, worst, err = run_cell(rob, lob, b)
        if err:
            fails += 1
            print(f"FAIL R/b={rob} lc/b={lob} b={b}: {err}")
            continue
        rows.append(row)
        print(f"ok R/b={rob:5.1f} lc/b={lob:4.2f} b={b}: worst {worst}%")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cells -> {out} ({fails} failed)")
    sys.exit(1 if fails or not rows else 0)


if __name__ == "__main__":
    main()

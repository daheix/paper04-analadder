#!/usr/bin/env python3
"""Paper04 B-ladder sweep: no-load airgap harmonics vs Carter-free analytic.

Sweeps slot_half_deg (slot opening) x lc_air on the 6-slot/2-pole sector.
Per cell: run_emag_getdp.py no-load solve, then extract from
emag_report.json: Bg1 (FE fundamental), Bg1_analytic_flat (Carter-free
1D magnetic circuit), err_vs_analytic_pct, Bg_pole_mean, THD_pct.

Expectation (RQ2): err grows with slot opening b0/ts — the Carter-free
anchor leaves its validity domain as slots open.

Writes data/sweep_b.csv.
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENG = REPO / "engine" / "simulation" / "getdp"
GETDP_BIN = "/home/wsc/wsc/ChinaSimStdio/bin/third_party/getdp/bin"
GMSH_BIN = "/home/wsc/wsc/ChinaSimStdio/bin/third_party/gmsh/bin"

SLOT_HALF = [1.0, 2.0, 3.0, 4.5, 6.0]   # deg; b0/ts = 0.033..0.20 at 6 slots
LC_AIR = [1.2e-3, 2.0e-3]               # grid axis (0.8e-3 fine only on 2 anchors)
FINE_ANCHORS = {1.0, 6.0}               # slot openings that also get lc=0.8e-3
BASE = {  # g2-6s2p baseline params (paper03 sample_regular family)
    "R_shaft": 0.01, "R_ri": 0.026, "mag_t": 0.003, "R_si": 0.03,
    "slot_depth": 0.012, "R_so": 0.05, "mag_half_deg": 76.5,
    "n_slots": 6, "n_poles": 2, "p": 1, "L_stack": 0.1, "Nc": 100.0,
    "magnetization": "radial", "halbach_segments": 1, "eccentricity_mm": 0.0,
}


def run_cell(sh, lc, wd):
    wd = Path(wd)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    p = dict(BASE, slot_half_deg=sh, lc_air=lc, lc_iron=1.2e-3, lc_shaft=2e-3)
    (wd / "params.json").write_text(json.dumps(p))
    env = {"PATH": f"{GMSH_BIN}:{GETDP_BIN}:/usr/bin:/bin", "HOME": "/home/wsc"}
    r = subprocess.run([sys.executable, "run_emag_getdp.py", "--params",
                        str(wd / "params.json"), "--workdir", str(wd)],
                       cwd=ENG, capture_output=True, text=True,
                       timeout=900, env=env)
    rep = wd / "emag_report.json"
    if r.returncode != 0 or not rep.exists():
        return None, f"solve fail rc={r.returncode}: {r.stderr[-150:]}"
    d = json.loads(rep.read_text())["no_load"]
    return {"slot_half_deg": sh, "lc_air_m": lc,
            "b0_over_ts": round(2 * 0.03 * __import__("math").sin(
                __import__("math").radians(sh)) / (2 * __import__("math").pi * 0.03 / 6), 4),
            "Bg1": d["Bg1"], "Bg1_analytic_flat": d["Bg1_analytic_flat"],
            "err_vs_analytic_pct": d["err_vs_analytic_pct"],
            "Bg_pole_mean": d["Bg_pole_mean"], "THD_pct": d["THD_pct"]}, None


def main():
    cells = [(sh, lc) for sh in SLOT_HALF for lc in LC_AIR]
    cells += [(sh, 0.8e-3) for sh in FINE_ANCHORS]
    rows = []
    for sh, lc in cells:
        wd = f"/tmp/p04_b_{sh}_{int(lc*1e4)}"
        row, err = run_cell(sh, lc, wd)
        if err:
            print(f"FAIL sh={sh} lc={lc}: {err}")
            continue
        rows.append(row)
        print(f"ok sh={sh:4.1f} b0/ts={row['b0_over_ts']:.3f} lc={lc}: "
              f"Bg1={row['Bg1']:.4f} flat={row['Bg1_analytic_flat']:.4f} "
              f"err={row['err_vs_analytic_pct']:.1f}%")
    out = REPO / "data" / "sweep_b.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cells -> {out}")
    sys.exit(0 if rows else 1)


if __name__ == "__main__":
    main()

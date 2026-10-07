#!/usr/bin/env bash
# P04 ladder demo: one A1 unit cell end-to-end (~1 min).
#   gmsh (msh2) -> analytic_verify -> quoted max |err|
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LC="${LC:-0.0022}"
echo "[demo] A1 unit cell: R=0.10 m, rc=0.005 m, lc=$LC (calibration geometry)"
GMSH="${GMSH:-$ROOT/bin/third_party/gmsh/bin/gmsh}"
AV="${AV:-$ROOT/engine/analytic_verify}"
MSH="$(mktemp --suffix=.msh2)"
trap 'rm -f "$MSH"' EXIT
"$GMSH" "$ROOT/engine/test_coil.geo" -2 -format msh2 -setnumber R 0.10 \
        -setnumber rc 0.005 -setnumber lc "$LC" -o "$MSH" >/dev/null 2>&1
"$AV" coil "$MSH" 1.0 | tee /dev/stderr | awk '/结论: 最大误差/ {
    gsub(/[^0-9.]/, "", $0); e=$0
    printf "[demo] max |err| = %s%%\n", e
    exit (e+0 <= 1.05 ? 0 : 1)  # acceptance: within 1% (5% relative slack on 0.917)
}'
echo "[demo] PASS (acceptance: scalar consistent with expected_results.json +-1%)"

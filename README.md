# AnalyticLadder (v0.1.0)

A three-rung analytic verification ladder for open-source 2-D magnetostatic
finite-element solvers, with coverage and failure domains. Companion software
for the SoftwareX submission (see the manuscript PDF).

## Requirements
- Python 3.11+
- Gmsh 4.15.2
- GetDP 4.0.0
- (optional) Docker, for the locked environment

## Quick start
```
make build    # build the locked Docker environment
make run      # run the preregistered 75-cell sweep
make verify   # numeric audit: every manuscript number vs released CSVs (29 checks)
```

One-command demo (about 1 minute): reproduces the calibration point 0.917%.
Expected outputs are frozen in `expected_results/` with SHA-256 hashes;
a re-run must match within 1%.

## Layout
- `engine/`            analytic ladder implementation (GetDP/Python)
- `scripts/`           sweep drivers
- `data/`              frozen CSV datasets (hashed)
- `expected_results/`  reference outputs + sha256
- `repro/`             one-command reproduction package
- `METADATA_TABLE.md`  completed SoftwareX metadata table

## Licence
MIT — see `LICENCE.txt`.

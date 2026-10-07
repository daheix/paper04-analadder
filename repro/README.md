# P04 repro package (L2)

Locked environment: python:3.12.3-slim-bookworm + exact pins
(repro/requirements.lock) + bundled Gmsh 4.15.2 / GetDP 3.5.

```
docker build -f repro/Dockerfile -t p04-ladder:locked .
docker run --rm p04-ladder:locked bash scripts/run_ladder_demo.sh   # run
docker run --rm p04-ladder:locked python3 scripts/verify_manuscript_numbers.py  # verify
```

Acceptance: verify exits 0 (29/29 checks on frozen data), demo prints
max |err| for the A1 unit cell consistent with
expected_results/expected_results.json within +-1%.

Honest TODO: `docker build` has not been executed on the package-creation
host (no container runtime); base-image digest to be back-filled after the
first pull on a docker-capable machine.

# P04 L2 reproducible package - analytic verification ladder.
# Base version-pinned; digest to be back-filled after first docker pull on a
# docker-capable host (build environment has no container runtime - honest
# TODO, see repro/README.md).
FROM python:3.12.3-slim-bookworm

# libblas3 (getdp runtime), build-essential+eigen3 (analytic_verify.cpp), make.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libblas3 make build-essential libeigen3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /p04

# Python layer - exact pins only.
COPY repro/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

# Bundled solvers (self-contained binaries).
COPY bin/third_party/ /p04/bin/third_party/
ENV PATH="/p04/bin/third_party/gmsh/bin:/p04/bin/third_party/getdp/bin:${PATH}"

# Engine snapshot (cpp + geo + getdp python chain), sweep scripts, frozen data.
COPY engine/ /p04/engine/
COPY scripts/ /p04/scripts/
COPY data/ /p04/data/
COPY expected_results/ /p04/expected_results/

# Compile the analytic verifier against the image's eigen3.
RUN g++ -O2 -I/usr/include/eigen3 /p04/engine/analytic_verify.cpp -o /p04/engine/analytic_verify

CMD ["python3", "scripts/verify_manuscript_numbers.py"]

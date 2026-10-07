# AnalyticLadder — root Makefile (delegates to repro/)
# Targets: make build | make run | make verify
# See README.md; the locked environment lives in repro/ (Dockerfile + requirements.lock).

.PHONY: build run verify demo

build:
	$(MAKE) -C repro build

run:
	$(MAKE) -C repro run

verify:
	$(MAKE) -C repro verify

demo:
	bash scripts/run_ladder_demo.sh

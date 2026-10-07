# P04 reproducible-package driver.
# Build context is the paper04_analadder repo root:
#   docker build -f repro/Dockerfile -t p04-ladder:locked .
SHELL := /bin/bash
IMAGE ?= p04-ladder:locked

.PHONY: build run verify help
build:            ## build the locked image
	docker build -f repro/Dockerfile -t $(IMAGE) .

run:              ## single-end-to-end ladder demo (A1 unit cell, ~1 min)
	docker run --rm $(IMAGE) bash scripts/run_ladder_demo.sh

verify:           ## 29-check manuscript-number audit on frozen data
	docker run --rm $(IMAGE) python3 scripts/verify_manuscript_numbers.py

help:
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | \
		awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

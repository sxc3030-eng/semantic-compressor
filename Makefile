# Makefile for Semantic Compressor.
#
# Targets wrap the most common commands so contributors do not need to
# remember exact pytest invocations. This Makefile is meant for Unix users
# (Linux / macOS). Windows users should use install.ps1 directly and run
# commands manually from an activated .venv.

.PHONY: help install test test-fast run-poc run-poc-aggressive fingerprint clean

help:
	@echo "Available targets:"
	@echo "  install            Create venv + install deps (runs install.sh)"
	@echo "  test               Run all 80 tests"
	@echo "  test-fast          Run only fast unit tests (skip integration)"
	@echo "  run-poc            Run the end-to-end POC demo"
	@echo "  run-poc-aggressive Run the end-to-end POC with --aggressive-uuid"
	@echo "  fingerprint        Generate visual fingerprints"
	@echo "  clean              Remove venv, caches, generated output"

install:
	@bash install.sh

test:
	.venv/bin/python -m pytest tests/ -v

test-fast:
	.venv/bin/python -m pytest tests/ -v -k "not real_users and not end_to_end"

run-poc:
	.venv/bin/python examples/run_poc.py

run-poc-aggressive:
	.venv/bin/python examples/run_poc.py --aggressive-uuid

fingerprint:
	.venv/bin/python examples/render_fingerprint.py

clean:
	rm -rf .venv .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
	rm -f output/recipes/users.md output/anchors/users_anchors.parquet
	rm -f data/reconstructed/users.csv
	rm -f output/profiling_reports/users.html
	rm -f output/fingerprints/*.png

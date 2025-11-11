# ---- Config ----
PY ?= 3.12               # default local Python for lint/typecheck
UV ?= uv
TOX_ENVS ?= py312,py313,py314

# Helpful env: force CPU unless you explicitly run GPU tests
export RAG_BENCH_DEVICE ?= cpu
export CUDA_VISIBLE_DEVICES ?=

# ---- Phony targets ----
.PHONY: help setup sync dev lint_n_type format test test-all test-py build clean distclean

help:
	@echo "Common tasks:"
	@echo "  make setup        Install uv (if needed) & tooling hints"
	@echo "  make sync         Create/refresh local venv with dev deps"
	@echo "  make dev          Sync + preflight (lint, typecheck, unit/offline tests)"
	@echo "  make lint         flake8 + isort --check + black --check"
	@echo "  make typecheck    mypy over src/"
	@echo "  make format       Apply isort + black"
	@echo "  make test         Run unit/offline tests on current Python"
	@echo "  make test-all     Run matrix tests via tox (py311/12/13)"
	@echo "  make test-py PY=3.11  Run tests using a specific Python"
	@echo "  make build        Build sdist+wheel"
	@echo "  make clean        Remove caches/build artefacts"
	@echo "  make distclean    Also remove venvs and tox envs"

setup:
	@command -v $(UV) >/dev/null || (echo "Installing uv..."; \
		curl -fsSL https://astral.sh/uv/install.sh | sh)
	@echo "uv installed: $$($(UV) --version)"

sync:
	$(UV) python install $(PY)
	$(UV) venv
	$(UV) sync --all-extras --dev

dev: sync lint typecheck test

lint_n_type:
	$(UV) run flake8
	$(UV) run isort --check-only .
	$(UV) run black --check .
	$(UV) run mypy .

format:
	$(UV) run isort .
	$(UV) run black .

test:
	$(UV) run pytest -q -m "unit or offline" --disable-warnings

# Run matrix locally via tox + tox-uv (no global installs)
test-all:
	$(UV) tool run --from tox-uv tox -e $(TOX_ENVS)

# Choose a Python version for tests quickly
test-py:
	$(UV) python install $(PY)
	PYTHON=$(PY) $(UV) run pytest -q -m "unit or offline" --disable-warnings

build:
	$(UV) run python -m build

clean:
	@rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
	@find . -type d -name "__pycache__" -exec rm -rf {} +

distclean: clean
	@rm -rf .venv .tox

# ---- Config ----
PY ?= 3.12
UVX ?= uvx
UV ?= $(UVX) uv
TOX_ENVS ?= py312,py313,py314
TOX_GPU_ENVS ?= gpupy312,gpupy313,gpupy314
PY_COMPACT := $(subst .,,$(PY))
TOX_PY_ENV ?= py$(PY_COMPACT)
TOX_CMD ?= $(UVX) --with tox-uv tox

# Default tests run on CPU for stability
export RAG_BENCH_DEVICE ?= cpu
export CUDA_VISIBLE_DEVICES ?=

# ---- Phony targets ----
.PHONY: help setup sync dev lint typecheck format test test-all test-py test-gpu build clean distclean \
        coverage coverage-xml coverage-report coverage-erase coveralls-upload-local

help:
	@echo "Common tasks:"
	@echo "  make setup          Install uv (if needed)"
	@echo "  make sync           Create/refresh local venv with dev deps"
	@echo "  make dev            Lint + typecheck + unit/offline tests"
	@echo "  make lint           flake8 + isort --check + black --check"
	@echo "  make typecheck      mypy over src/"
	@echo "  make format         Apply isort + black"
	@echo "  make test           Unit/offline tests via tox for PY=$(PY)"
	@echo "  make test-all       Matrix tests via tox (py312/13/14)"
	@echo "  make test-gpu       GPU-marked tests via tox"
	@echo "  make build          Build sdist + wheel"
	@echo "  make coverage       Run tests and produce combined coverage report"
	@echo "  make clean          Remove caches/build artefacts"
	@echo "  make distclean      Also remove venvs and tox envs"

setup:
	@command -v $(UVX) >/dev/null || (echo "Installing uv..."; \
		curl -fsSL https://astral.sh/uv/install.sh | sh)
	@echo "uvx available: $$($(UVX) --version)"
	@echo "uv available via uvx: $$($(UV) --version)"

sync:
	$(UV) python install $(PY)
	$(UV) venv
	$(UV) sync --all-extras --dev

dev: sync lint typecheck test

lint:
	$(TOX_CMD) -e lint

typecheck:
	$(TOX_CMD) -e typecheck

format:
	$(UV) run isort .
	$(UV) run black .

test:
	$(UV) python install $(PY)
	$(TOX_CMD) -e $(TOX_PY_ENV)

test-all:
	$(TOX_CMD) -e $(TOX_ENVS)

test-all-gpu:
	$(TOX_CMD) -e $(TOX_GPU_ENVS)

build:
	$(UV) run python -m build

# -------- Coverage & Coveralls --------

coverage-erase:
	$(UV) run coverage erase

coverage:
	$(UV) run coverage erase
	$(TOX_CMD) -e py312
	$(TOX_CMD) -e gpupy312
	$(UV) run coverage combine
	$(UV) run coverage report
	@echo "Use 'make coverage-xml' for XML (Coveralls)"

coverage-xml-html:
	$(UV) run coverage xml
	$(UV) run coverage html
	@echo "Wrote coverage.xml and coverage.html"

clean:
	@rm -rf .pytest_cache .mypy_cache dist build coverage.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} +

distclean: clean
	@rm -rf .venv .tox

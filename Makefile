
.PHONY: install run test fmt lint uv-sync uv-run

install:
	python -m pip install -U pip
	pip install -e ".[dev]"

run:
	python run.py --config configs/wiki.yaml --question "What is LangChain?"

test:
	pytest -q

fmt:
	black .
	isort .

lint:
	flake8
	isort --check-only .
	black --check .

uv-sync:
	uv venv || true
	. .venv/bin/activate && uv sync

uv-run:
	. .venv/bin/activate && python run.py --config configs/wiki.yaml --question "What is LangChain?"

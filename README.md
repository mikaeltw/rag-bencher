
# rag-bench

[![CI](https://img.shields.io/github/actions/workflow/status/mikaeltw/rag-bench/ci.yml?branch=main)](../../actions)
[![PyPI](https://img.shields.io/pypi/v/rag-bench.svg)](https://pypi.org/project/rag-bench/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)



## Install

From PyPI (after first release):
```bash
pip install rag-bench
# or with uv
uv pip install rag-bench
```

CLI entrypoints:
```bash
rag-bench --config configs/wiki.yaml --question "What is LangChain?"
rag-bench-bench --config configs/multi_query.yaml --qa examples/qa/toy.jsonl
rag-bench-many --configs "configs/*.yaml" --qa examples/qa/toy.jsonl --dataset docs/wiki
```


Reproducible **RAG** baselines + evaluations, powered by **LangChain**. Configure a pipeline in YAML, run one command, get an HTML/terminal report.

## Quickstart (with `uv`)

1) Install uv: https://docs.astral.sh/uv/getting-started/
2) Create & activate a venv and sync deps:
```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync                     # resolves & installs according to pyproject
cp .env.example .env        # set your keys
```
3) Run:
```bash
python run.py --config configs/wiki.yaml --question "What is LangChain?"
```

> Tip: set a model via env expansion: `MODEL_NAME=gpt-4o-mini` (see `configs/wiki.yaml`).

## Plain pip alternative

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python run.py --config configs/wiki.yaml --question "What is LangChain?"
```

## Why

Everyone builds RAG; few can **compare** them cleanly. `rag-bench` gives you:
- Strict, validated configs (Pydantic) with env var expansion
- Plug-and-play baselines (naive RAG, multi-query, HyDE, rerank-ready)
- Simple datasets and metrics (faithfulness, answer relevancy, context recall - stubs included)
- HTML/CLI reports
- Extensible pipeline interface

## Repo Layout

```
rag_bench/
  pipelines/         # RAG variants
  eval/              # datasets + metrics + report
  utils/             # logging, io, helpers
  config.py          # strict config schema/loader
configs/             # YAML configs
examples/            # minimal dataset & notebook
reports/             # generated reports
tests/               # unit tests
```

## Contributing

- Good first issues: add a new retriever / metric / dataset loader
- Please run `ruff` and `pytest` before pushing
- MIT license


## Multi-query baseline

Run:
```bash
python run.py --config configs/multi_query.yaml --question "What is LangChain?"
```
The report includes generated sub-queries and retrieved snippets for transparency.


## Rerank baseline

Try a rerank stage (cross-encoder if available, otherwise cosine fallback):

```bash
python run.py --config configs/rerank.yaml --question "What is LangChain?"
```

The report will include a **Rerank candidates** table with scores.


## Evaluation harness

Run a small benchmark on a toy QA set:
```bash
python bench.py --config configs/multi_query.yaml --qa examples/qa/toy.jsonl
```
Outputs per-question metrics (lexical F1, bag-of-words cosine, context recall) and an HTML summary in `reports/`.


## Datasets registry
Place raw text in `examples/datasets/<group>/<name>/*.txt|*.md`. Load via:
```bash
python run.py --config configs/wiki.yaml --dataset docs/wiki --question "What is LangChain?"
```

## Caching & reproducibility
- On-disk cache of answers in `.ragbench_cache/`
- `--seed` and `--run-id` switches in `run.py`
- Approx token/cost tracking via a LangChain callback (set `--cost-in/--cost-out` to compute rough $)

## HyDE baseline
```bash
python run.py --config configs/hyde.yaml --question "What is LangChain?"
```

## Multi-run comparison
```bash
python bench_many.py --configs "configs/*.yaml" --qa examples/qa/toy.jsonl --dataset docs/wiki
```
Produces a sortable HTML summary in `reports/summary-*.html`.


### Packaged resources
If installed from PyPI, example files are bundled. You can locate them via Python:
```python
from rag_bench.utils.resources import get_resource_path
print(get_resource_path("examples/data/sample.txt"))
print(get_resource_path("examples/qa/toy.jsonl"))
print(get_resource_path("configs/wiki.yaml"))
```
Use these absolute paths with the CLI if you want to run examples without cloning.

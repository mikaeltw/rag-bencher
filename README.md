<p align="center">
  <img src="https://github.com/mikaeltw/rag-bencher/raw/main/assets/rag-bencher-logo.png" alt="rag-bench logo" width="820"/>
</p>

<p align="center">
  <b>Reproducible, extensible, multi-provider benchmarking for Retrieval-Augmented Generation (RAG) systems.</b>
</p>

![Python Versions](https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-blue?logo=python)
[![CI](https://github.com/mikaeltw/rag-bencher/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mikaeltw/rag-bencher/actions/workflows/ci.yml)
[![Lint & Typecheck](https://github.com/mikaeltw/rag-bencher/actions/workflows/lint-type-check.yml/badge.svg?branch=main)](https://github.com/mikaeltw/rag-bencher/actions/workflows/lint-type-check.yml)
[![Build & Test release](https://github.com/mikaeltw/rag-bencher/actions/workflows/release-test.yml/badge.svg?branch=main)](https://github.com/mikaeltw/rag-bencher/actions/workflows/release-test.yml)
[![Coverage Status](https://img.shields.io/coveralls/github/mikaeltw/rag-bencher?branch=main&logo=coveralls)](https://coveralls.io/github/mikaeltw/rag-bencher?branch=main)
[![PyPI](https://img.shields.io/pypi/v/rag-bencher.svg?logo=pypi&flat)](https://pypi.org/project/rag-bencher/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?logo=open-source-initiative&logoColor=white)](LICENSE)
![Repo size](https://img.shields.io/github/repo-size/mikaeltw/rag-bencher?logo=github&logoColor=white)
![Code size](https://img.shields.io/github/languages/code-size/mikaeltw/rag-bencher?logo=codefactor&logoColor=white)

---


Reproducible retrieval-augmented generation (RAG) baselines and an evaluation harness that makes it easy to compare pipelines across providers. Configure a pipeline in YAML, run one command, and collect HTML reports that stay consistent across experiments.

## What "RAG comparison" means here
- Each pipeline is defined by a YAML file that describes the chat model, embeddings, retriever settings, and optional provider/vector adapters.
- The CLI runs the pipeline over a QA set and reports lexical F1, bag-of-words cosine, and context recall.
- Multi-run mode sweeps multiple configs to show relative performance on the same dataset.
- Everything is reproducible: configs are validated, caching is on by default, and offline CPU runs are available.

## Features
- Ready-to-run pipelines: naive, multi-query, HyDE, and rerank.
- Config-first workflow with strict validation (Pydantic) and reproducible defaults.
- Optional adapters for OpenAI-compatible APIs, Azure OpenAI, AWS Bedrock, Vertex AI, Azure AI Search, OpenSearch, and Matching Engine.
- HTML reports for single runs and multi-run comparisons.
- Works fully offline on CPU via bundled Hugging Face models when cloud access is not available.

## Installation
Requirements: Python 3.12–3.14.

### PyPI
```bash
pip install rag-bencher
# or using uv
uv pip install rag-bencher
```

### Provider and vector extras
```bash
pip install "rag-bencher[gcp]"    # Vertex AI chat + Matching Engine
pip install "rag-bencher[aws]"    # Bedrock chat + OpenSearch vector
pip install "rag-bencher[azure]"  # Azure OpenAI chat + Azure AI Search
pip install "rag-bencher[providers]"  # installs all provider extras
```

### From source (development)
```bash
git clone https://github.com/mikaeltw/rag-bencher.git
cd rag-bencher
python -m venv venv && source venv/bin/activate
pip install -e .[dev]
# or use uv + tox via Makefile helpers
(make setup)         # Optional for installing uv
make sync            # create/refresh uv-managed venv with dev extras
make dev             # lint + typecheck + tests
```
The helper targets download dependencies; populate wheels locally if your network is restricted.

## Quickstart

### Ask a single question
```bash
python scripts/run.py --config configs/wiki.yaml --question "What is LangChain?"
# or, when installed as a package
python -m rag_bencher.cli --config configs/wiki.yaml --question "What is LangChain?"
```

### Compare two configs via CLI
```bash
python -m rag_bencher.bench_many_cli \
  --configs "configs/*.yaml" \
  --qa examples/qa/toy.jsonl
```
Generates an HTML summary under `reports/summary-*.html` so you can scan relative scores quickly.

### Minimal Python comparison example
```python
from pathlib import Path

from rag_bencher.config import load_config
from rag_bencher.eval.dataset_loader import load_texts_as_documents
from rag_bencher.eval.metrics import bow_cosine, context_recall, lexical_f1
from rag_bencher.pipelines.selector import select_pipeline

qa_examples = [
    {"question": "What is LangChain?", "reference_answer": "A Python framework for building LLM apps."},
    {"question": "What is RAG?", "reference_answer": "Retrieval-augmented generation combines search and generation."},
]

docs = load_texts_as_documents(["examples/data/sample.txt"])


def evaluate(config_path: str) -> tuple[str, dict[str, float]]:
    selection = select_pipeline(config_path, docs, load_config(config_path))
    chain = selection.chain
    debug = selection.debug

    scores: list[dict[str, float]] = []
    for qa in qa_examples:
        answer = chain.invoke(qa["question"])
        retrieved = ""
        dbg = debug()
        if dbg.get("retrieved"):
            retrieved = "\n".join(item.get("preview", "") for item in dbg["retrieved"])
        elif dbg.get("candidates"):
            retrieved = "\n".join(item.get("preview", "") for item in dbg["candidates"][:5])
        scores.append(
            {
                "lexical_f1": lexical_f1(answer, qa["reference_answer"]),
                "bow_cosine": bow_cosine(answer, qa["reference_answer"]),
                "context_recall": context_recall(qa["reference_answer"], retrieved) if retrieved else 0.0,
            }
        )

    avg = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0].keys()}
    return selection.pipeline_id, avg


for cfg in ("configs/wiki.yaml", "configs/rerank.yaml"):
    pid, metrics = evaluate(cfg)
    print(f"{Path(cfg).name} ({pid}) -> {metrics}")
```

## Examples
- `examples/compare_two_pipelines.py`: small script to score two configs on a tiny QA set.
- `examples/quickstart.ipynb`: walkthrough notebook.
- Sample corpus lives in `examples/data/` and QA fixtures in `examples/qa/`.

## Running tests
- `make test` runs offline/unit tests via tox for the configured Python version.
- `make test-all` runs the full matrix (py312/py313/py314).
- `make test-all-gpu` covers GPU-marked tests on a GPU host.
- `make dev` runs lint, typecheck, and tests together.

## Requirements and dependencies
- Core dependencies: LangChain, LangChain Community/OpenAI/Hugging Face adapters, sentence-transformers.
- Extras pull in provider-specific SDKs (Azure, AWS, GCP) and vector backends.
- Optional: `uv` for reproducible installs, `make` targets for day-to-day tasks.

## Configuration basics
- All runtime behavior is defined in YAML (see `configs/`).
- Common keys: `model`, `retriever`, `data.paths`, `runtime.offline`, `provider` (chat/embedding adapters), and `vector` (alternate retriever stores).
- Pipelines are toggled by presence of `multi_query`, `hyde`, or `rerank`; absence defaults to the naive RAG pipeline.
- Provider configs pull credentials from environment variables; see `configs/providers/` for examples.

## Architecture overview
- **CLI entrypoints:** `rag_bencher.cli` (single question), `rag_bencher.bench_cli` (single-config benchmarking), `rag_bencher.bench_many_cli` (multi-config comparisons).
- **Config layer:** `rag_bencher.config.BenchConfig` validates YAML and wires provider/vector extras.
- **Pipeline builders:** `rag_bencher.pipelines.*` assemble LangChain runnables for naive, multi-query, HyDE, and rerank flows.
- **Evaluation:** `rag_bencher.eval.*` loads datasets, computes metrics, and writes HTML reports.
- **Providers/vectors:** adapters under `rag_bencher.providers` and `rag_bencher.vector` wrap cloud services while preserving the same interface.
- **Reproducibility:** caches answers in `.ragbencher_cache/`, sets seeds, and keeps reports in `reports/`.

## Roadmap / future work
- Add more provider smoke tests and CI examples.
- Ship richer metrics (cost, latency) alongside the existing text similarity scores.
- Expand notebooks and end-to-end examples for custom corpora.

## License
MIT License; see `LICENSE`.

## Security and support
- Vulnerability handling is described in `SECURITY.md`.
- Code of conduct: `CODE_OF_CONDUCT.md`.
- Open an issue or discussion on GitHub for questions and feature requests.

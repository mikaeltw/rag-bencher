
# rag-bench

[![CI](https://img.shields.io/github/actions/workflow/status/mikaeltw/rag-bench/ci.yml?branch=main)](../../actions)
[![PyPI](https://img.shields.io/pypi/v/rag-bench.svg)](https://pypi.org/project/rag-bench/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Reproducible retrieval-augmented generation (**RAG**) baselines and evaluation tools built on **LangChain**. Configure a pipeline in YAML, run a single command, and collect HTML reports that are easy to compare across experiments.

---

## Why rag-bench?
- Batteries-included baseline pipelines: **naive**, **multi-query**, **HyDE**, and **rerank**.
- Config-first workflow with strict validation (Pydantic) and reproducible defaults.
- First-class support for offline CPU runs, OpenAI-compatible chat models, and managed vector stores.
- Evaluation harness that produces HTML summaries, ready for sharing.
- CI-friendly: linting, smoke tests against cloud providers, GPU checks, and publishing workflows.

---

## Installation

### PyPI
```bash
pip install rag-bench
# or using uv
uv pip install rag-bench
```

### Provider & vector extras
```bash
pip install "rag-bench[gcp]"    # Google Vertex AI chat + Matching Engine
pip install "rag-bench[aws]"    # Bedrock chat + OpenSearch vector
pip install "rag-bench[azure]"  # Azure OpenAI chat + Azure AI Search vector
pip install "rag-bench[providers]"  # install all provider extras
```

### From source (development)
```bash
git clone https://github.com/mikaeltw/rag-bench.git
cd rag-bench
python -m venv venv && source venv/bin/activate
make install          # installs rag-bench in editable mode with dev extras
```
> The `make install` target reaches out to PyPI; if your environment blocks network access you will need to pre-populate wheels manually.

---

## Quick Start

### Ask a single question
```bash
# From a cloned repo
python run.py --config configs/wiki.yaml --question "What is LangChain?"

# When installed as a package
python -m rag_bench.cli --config configs/wiki.yaml --question "What is LangChain?"
```

### Switch pipelines
```bash
python -m rag_bench.cli --config configs/multi_query.yaml --question "What is LangChain?"
python -m rag_bench.cli --config configs/rerank.yaml --question "What is LangChain?"
python -m rag_bench.cli --config configs/hyde.yaml --question "What is LangChain?"
```

### Run fully offline on CPU
```bash
RAG_BENCH_DEVICE=cpu \
python -m rag_bench.cli --config configs/wiki_offline.yaml --question "What is LangChain?"
```
Offline mode uses the `google/flan-t5-small` seq2seq model via Hugging Face for deterministic answers on CPUs.

### Target a managed provider
```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
python -m rag_bench.cli --config configs/providers/azure.yaml --question "What is LangChain?"
```
Every provider config expects credentials via environment variables; see [Configuration Reference](#configuration-reference).

---

## Evaluation Harness

### Benchmark one pipeline against a QA set
```bash
python -m rag_bench.bench_cli \
  --config configs/multi_query.yaml \
  --qa examples/qa/toy.jsonl
```
- Streams per-example metrics to the terminal.
- Computes average `lexical_f1`, `bow_cosine`, and `context_recall`.
- Emits a timestamped HTML report under `reports/`.

### Compare multiple configs side-by-side
```bash
python -m rag_bench.bench_many_cli \
  --configs "configs/*.yaml" \
  --qa examples/qa/toy.jsonl
```
Produces an aggregated HTML table (`reports/summary-*.html`) so you can compare pipelines across the same dataset quickly.

---

## Configuration Reference

All runtime behaviour is described through YAML files. The schema is enforced by `rag_bench.config.BenchConfig`:

```yaml
model:
  name: gpt-4o-mini
retriever:
  k: 4
data:
  paths:
    - examples/data/sample.txt
runtime:
  offline: false          # switch to true for CPU Hugging Face runs
  device: auto            # auto | cpu | cuda
provider:
  name: azure             # optional; see provider examples below
  chat:
    deployment: gpt-4o-mini
    endpoint: ${AZURE_OPENAI_ENDPOINT}
vector:
  name: azure_ai_search   # optional; see vector examples below
  endpoint: https://<>.search.windows.net
```

### Pipelines
Each pipeline is enabled by a top-level key:

| Pipeline | Config file | Additional key(s) |
|----------|-------------|-------------------|
| Naive retrieval (default) | `configs/wiki.yaml` | none |
| Multi-query | `configs/multi_query.yaml` | `multi_query.n_queries` |
| HyDE | `configs/hyde.yaml` | `hyde` block (empty is fine) |
| Rerank | `configs/rerank.yaml` | `rerank.method`, `rerank.top_k`, optional `cross_encoder_model` |

### Provider adapters
| Provider | Config | Required environment variables |
|----------|--------|--------------------------------|
| AWS Bedrock | `configs/providers/aws.yaml` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, optional `BEDROCK_MODEL` |
| Azure OpenAI | `configs/providers/azure.yaml` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, optional `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` |
| Google Vertex | `configs/providers/gcp.yaml` | `GCP_PROJECT`, `VERTEX_LOCATION`, optional `VERTEX_CHAT_MODEL`, `GOOGLE_APPLICATION_CREDENTIALS` |

If `provider` is omitted the CLI defaults to `langchain_openai.ChatOpenAI` and uses the API key in your environment.

### Vector backends
Supply a `vector` block to replace the default FAISS retriever:

```yaml
vector:
  name: azure_ai_search
  endpoint: https://<your>.search.windows.net
  index: rag-bench
  api_key: ${AZURE_SEARCH_API_KEY}
```

Available adapters:
- `azure_ai_search`: requires `azure-search-documents` (install `rag-bench[azure]`).
- `opensearch`: configure `hosts`, `index`, and either HTTP basic auth or IAM.
- `matching_engine`: requires `project_id`, `location`, `index_id`, `endpoint_id` (install `rag-bench[gcp]`).

### Datasets
- `data.paths` accepts one or more plain-text files. Use `examples/data/sample.txt` as a template.
- For question/answer benchmarking, use JSONL (`{"question": "...", "reference_answer": "..."}`) files such as `examples/qa/toy.jsonl`.

### Runtime options
- `runtime.offline: true` switches to the bundled Hugging Face pipeline.
- `runtime.device`: `"auto"` (GPU if available), `"cpu"`, or `"cuda"`.
- `RAG_BENCH_DEVICE` environment variable overrides the device across modules.

---

## Reports, Caching, and Reproducibility
- Reports are saved under `reports/` with timestamps (see `rag_bench.eval.report`).
- Answers are cached per model/question under `.ragbench_cache/` to avoid re-billing cloud LLMs.
- `rag_bench.utils.repro.set_seeds(42)` is invoked by the CLI to keep vector splits deterministic.

---

## Development & Contribution
- Follow the guidance in [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, formatting, linting, and tests.
- Run `make fmt && make lint && make test` before pushing.
- Update [CHANGELOG.md](CHANGELOG.md) for user-facing changes and [RELEASE.md](RELEASE.md) ahead of publishing.
- Branch protection recommendations live in `docs/branch_protection.md`.

---

## Continuous Integration

![CI](https://github.com/mikaeltw/rag-bench/actions/workflows/ci.yml/badge.svg)
![Nightly Cloud](https://github.com/mikaeltw/rag-bench/actions/workflows/live-cloud.yml/badge.svg)
![GPU Tests](https://github.com/mikaeltw/rag-bench/actions/workflows/gpu.yml/badge.svg)

| Workflow | Trigger | Highlights |
|----------|---------|------------|
| `ci.yml` | push / PR | Linting, formatting checks, unit & offline tests |
| `live-cloud.yml` | nightly + manual | Vertex AI smoke tests |
| `gpu.yml` | manual | GPU-tagged tests on self-hosted runners |
| `publish-testpypi.yml` / `publish-pypi.yml` | GitHub releases | Build + publish wheels |

Secrets required by each job are summarised in the workflow files; mirror them in your repository settings.

---

## Support & Security
- Vulnerabilities: follow the process in [SECURITY.md](SECURITY.md) (private advisory, response within 7 days).
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Have questions or ideas? Open an issue or discussion on GitHub. Happy benchmarking!

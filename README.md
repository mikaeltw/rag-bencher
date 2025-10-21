
# rag-bench

[![CI](https://img.shields.io/github/actions/workflow/status/mikaeltw/rag-bench/ci.yml?branch=main)](../../actions)
[![PyPI](https://img.shields.io/pypi/v/rag-bench.svg)](https://pypi.org/project/rag-bench/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Reproducible **RAG** baselines + evaluations, powered by **LangChain**. Configure a pipeline in YAML, run one command, get an HTML report.

## Install
```bash
pip install rag-bench
# or
uv pip install rag-bench
```

## Quickstart
```bash
python run.py --config configs/wiki.yaml --question "What is LangChain?"
python run.py --config configs/multi_query.yaml --question "What is LangChain?"
python run.py --config configs/rerank.yaml --question "What is LangChain?"
python run.py --config configs/hyde.yaml --question "What is LangChain?"
```

## Cloud adapters (optional)
```bash
pip install "rag-bench[gcp]"    # Vertex AI
pip install "rag-bench[aws]"    # Bedrock
pip install "rag-bench[azure]"  # Azure OpenAI
# Then:
rag-bench --config configs/providers/azure.yaml --question "What is LangChain?"
```

## Eval harness
```bash
rag-bench-bench --config configs/multi_query.yaml --qa examples/qa/toy.jsonl
rag-bench-many --configs "configs/*.yaml" --qa examples/qa/toy.jsonl
```

## Contributing
- `make fmt && make lint && make test`
- Open PRs against `main`

See `RELEASE.md` for publishing steps.

### Vector backends (optional)

Use a managed vector store instead of local FAISS by adding a `vector:` block:

```yaml
vector:
  name: azure_ai_search
  endpoint: https://<your>.search.windows.net
  index: my-index
  # api_key: ${AZURE_SEARCH_API_KEY}
```

Other options:
- `opensearch`: `hosts`, `index` (plus IAM/http_auth).
- `matching_engine` (GCP): `project_id`, `location`, `index_id`, `endpoint_id`.

Install extras:
```bash
pip install "rag-bench[azure]"  # for Azure AI Search
pip install "rag-bench[aws]"    # for OpenSearch
pip install "rag-bench[gcp]"    # for Matching Engine
```

## 🧪 Continuous Integration

![CI](https://github.com/mikaeltw/rag-bench/actions/workflows/ci.yml/badge.svg)
![Nightly Cloud](https://github.com/mikaeltw/rag-bench/actions/workflows/live-cloud.yml/badge.svg)
![GPU Tests](https://github.com/mikaeltw/rag-bench/actions/workflows/gpu.yml/badge.svg)

CI runs:
- ✅ Unit & Offline tests on every push / PR
- ☁️ Cloud/vector smokes on push — **allowed to fail**
- ⚙️ Nightly deeper tests in `.github/workflows/live-cloud.yml`
- 🧩 GPU tests on self-hosted runners via `.github/workflows/gpu.yml`

### 🔐 Secrets & Variables

| Job | Required Secrets / Variables | Purpose |
|------|-------------------------------|----------|
| **cloud-gcp** | `GOOGLE_CREDENTIALS_JSON`, `GCP_PROJECT`, `VERTEX_LOCATION`, `VERTEX_CHAT_MODEL` | Vertex AI chat smoke test |
| **cloud-aws** | `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `BEDROCK_MODEL` | AWS Bedrock chat smoke test |
| **cloud-azure** | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` | Azure OpenAI chat smoke test |
| **vector-smokes** | `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_INDEX`, `OPENSEARCH_HOST`, `OPENSEARCH_INDEX`, `ME_INDEX_ID`, `ME_ENDPOINT_ID`, `VERTEX_LOCATION` | Vector store backends |

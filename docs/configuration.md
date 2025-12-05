# Configuration guide

All runtime behavior is described through YAML files in `configs/`. Configs are validated by `rag_bencher.config.BenchConfig` to keep experiments reproducible.

## Core fields
```yaml
model:
  name: gpt-4o-mini
retriever:
  k: 4
data:
  paths:
    - examples/data/sample.txt
runtime:
  offline: false      # switch to true for CPU-only Hugging Face runs
  device: auto        # auto | cpu | cuda
```

## Pipelines
Enable a pipeline by including one of these blocks:
- `multi_query`: sets `n_queries` for query expansion.
- `hyde`: toggles HyDE synthetic queries.
- `rerank`: set `method`, `top_k`, and optional `cross_encoder_model`.
If none are present, the naive retriever pipeline is used.

## Providers
Add a `provider` block to replace the default OpenAI-compatible chat model:
```yaml
provider:
  name: azure
  chat:
    deployment: gpt-4o-mini
    endpoint: ${AZURE_OPENAI_ENDPOINT}
```
Supported adapters live in `configs/providers/`. Credentials are read from the environment.

## Vector backends
Use `vector` to swap out the FAISS retriever:
```yaml
vector:
  name: azure_ai_search
  endpoint: https://<your>.search.windows.net
  index: rag-bencher
  api_key: ${AZURE_SEARCH_API_KEY}
```
Adapters exist for Azure AI Search, OpenSearch, and Matching Engine; extra dependencies are pulled in via the matching extras.

## Tips
- Keep config filenames descriptive (pipeline + provider), e.g., `hyde_azure.yaml`.
- Store small sample corpora under `examples/data/` and QA sets under `examples/qa/` for repeatable runs.
- Prefer environment variables for secrets; never commit credentials to configs.

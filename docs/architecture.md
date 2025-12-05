# Architecture

This project keeps RAG pipelines configurable while sharing evaluation code across providers and vector backends.

## Components
- **CLI**: `rag_bencher.cli` answers a single question, `rag_bencher.bench_cli` benchmarks one config, and `rag_bencher.bench_many_cli` compares multiple configs and writes a summary report.
- **Configuration**: `rag_bencher.config.BenchConfig` validates YAML files, applies defaults, and wires optional provider/vector adapters.
- **Pipelines**: Builders in `rag_bencher.pipelines` assemble LangChain runnables for naive, multi-query, HyDE, and rerank flows and expose a debug hook to inspect retrieval.
- **Providers and vectors**: Adapters in `rag_bencher.providers` and `rag_bencher.vector` wrap cloud chat/embedding APIs and managed vector stores while keeping the interface consistent.
- **Evaluation**: `rag_bencher.eval` loads corpora, runs QA datasets, computes metrics, and writes HTML reports for single and multi-run workflows.
- **Reproducibility**: deterministic seeds, `.ragbencher_cache/` for answer caching, and timestamped reports under `reports/`.

## Data flow
1. Load YAML config with `rag_bencher.config.load_config`.
2. Convert text sources into `Document` objects via `rag_bencher.eval.dataset_loader`.
3. Select a pipeline with `rag_bencher.pipelines.selector.select_pipeline`, which builds the runnable chain and debug hook.
4. Invoke the chain for each question, compute metrics (lexical F1, bag-of-words cosine, context recall), and collect results.
5. Emit an HTML report with configuration metadata for reproducibility.

## Extending safely
- Add new pipelines under `src/rag_bencher/pipelines/` and surface them via `select_pipeline`.
- Keep adapters small and composable; prefer optional dependencies behind extras.
- Use offline-friendly fixtures in tests and gate GPU-heavy paths with markers.

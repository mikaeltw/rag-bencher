
import argparse, os, yaml
from rich.console import Console
from rag_bench.config import load_config
from rag_bench.eval.dataset_loader import load_texts_as_documents
from rag_bench.eval.datasets import load_dataset
from rag_bench.eval.report import write_simple_report
from rag_bench.utils.repro import set_seeds, make_run_id
from rag_bench.utils.callbacks.usage import UsageTracker
from rag_bench.utils.cache import cache_get, cache_set

from rag_bench.pipelines import naive_rag
from rag_bench.pipelines import multi_query as mq
from rag_bench.pipelines import rerank as rr
from rag_bench.pipelines import hyde as hy

console = Console()

def _choose_pipeline(cfg_path: str, docs):
    cfg = load_config(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(os.path.expandvars(f.read())) or {}
    if "rerank" in raw_cfg:
        rrc = raw_cfg["rerank"]
        chain, debug = rr.build_chain(
            docs, model=cfg.model.name, k=cfg.retriever.k,
            rerank_top_k=int(rrc.get("top_k", 4)),
            method=str(rrc.get("method", "auto")),
            cross_encoder_model=str(rrc.get("cross_encoder_model", "BAAI/bge-reranker-base")),
        )
        return "rerank", chain, debug, cfg
    if "multi_query" in raw_cfg:
        n_queries = int(raw_cfg["multi_query"].get("n_queries", 3))
        chain, debug = mq.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, n_queries=n_queries)
        return "multi_query", chain, debug, cfg
    if "hyde" in raw_cfg:
        chain, debug = hy.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k)
        return "hyde", chain, debug, cfg
    chain, debug = naive_rag.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k)
    return "naive", chain, debug, cfg

def main():
    parser = argparse.ArgumentParser(description="Run rag-bench pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--dataset", help="Dataset name under examples/datasets (e.g., docs/wiki)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--cost-in", type=float, default=0.0, help="Approx $ per 1k input tokens")
    parser.add_argument("--cost-out", type=float, default=0.0, help="Approx $ per 1k output tokens")
    args = parser.parse_args()

    set_seeds(args.seed)
    run_id = args.run_id or make_run_id()

    # Load docs either from dataset registry or config data.paths
    if args.dataset:
        docs = load_dataset(args.dataset)
    else:
        cfg_probe = load_config(args.config)
        docs = load_texts_as_documents(cfg_probe.data.paths)

    pipe_id, chain, debug, cfg = _choose_pipeline(args.config, docs)

    # Add usage tracking callback
    tracker = UsageTracker(cost_per_1k_input=args.cost_in, cost_per_1k_output=args.cost_out)

    console.rule(f"[bold]Running pipeline [{pipe_id}] (run_id={run_id})")
    prompt = args.question
    cached = cache_get(cfg.model.name, prompt)
    if cached is not None:
        answer = cached
        console.print("[yellow]Loaded answer from cache[/yellow]")
    else:
        # Use LangChain run with callback for usage tracking
        answer = chain.invoke(prompt, config={"callbacks": [tracker]})
        cache_set(cfg.model.name, prompt, answer)

    console.print(f"[bold]Answer:[/bold] {answer}")
    extras = debug()
    usage = tracker.summary()
    extras["usage"] = usage
    extras["run_id"] = run_id
    cfg_dict = cfg.model_dump()
    report_path = write_simple_report(question=args.question, answer=answer, cfg=cfg_dict, extras=extras)
    console.print(f"[green]Report written to {report_path}[/green]")

if __name__ == "__main__":
    main()

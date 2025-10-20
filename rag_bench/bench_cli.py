
import argparse, os, json, yaml
from pathlib import Path
from statistics import mean
from rich.console import Console

from rag_bench.config import load_config
from rag_bench.eval.dataset_loader import load_texts_as_documents
from rag_bench.eval.metrics import lexical_f1, bow_cosine, context_recall
from rag_bench.eval.report import write_simple_report
from rag_bench.pipelines import naive_rag
from rag_bench.pipelines import multi_query as mq
from rag_bench.pipelines import rerank as rr

console = Console()

def _choose_pipeline(cfg_path: str, docs):
    cfg = load_config(cfg_path)
    import os, yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(os.path.expandvars(f.read())) or {}
    if "rerank" in raw_cfg:
        rrc = raw_cfg["rerank"]
        chain, debug = rr.build_chain(
            docs,
            model=cfg.model.name,
            k=cfg.retriever.k,
            rerank_top_k=int(rrc.get("top_k", 4)),
            method=str(rrc.get("method", "auto")),
            cross_encoder_model=str(rrc.get("cross_encoder_model", "BAAI/bge-reranker-base")),
        )
        pipe_id = "rerank"
    elif "multi_query" in raw_cfg:
        n_queries = int(raw_cfg["multi_query"].get("n_queries", 3))
        chain, debug = mq.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, n_queries=n_queries)
        pipe_id = "multi_query"
    else:
        chain, debug = naive_rag.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k)
        pipe_id = "naive"
    return cfg, chain, debug, pipe_id

def main():
    ap = argparse.ArgumentParser(description="Evaluate a RAG pipeline on a QA set")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--qa", required=True, help="Path to JSONL QA set with fields: question, reference_answer")
    args = ap.parse_args()

    # docs come from the config
    cfg = load_config(args.config)
    docs = load_texts_as_documents(cfg.data.paths)

    # construct chain + debug callback
    cfg2, chain, debug, pipe_id = _choose_pipeline(args.config, docs)

    rows = []
    with open(args.qa, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            ref = ex["reference_answer"]
            ans = chain.invoke(q)
            dbg = debug()

            # build a retrieved_text aggregate if available in debug info
            retrieved_text = ""
            if dbg.get("retrieved"):
                retrieved_text = "\n".join(r.get("preview","") for r in dbg["retrieved"])
            elif dbg.get("candidates"):
                # take top 5 candidates' previews
                top = dbg["candidates"][:5]
                retrieved_text = "\n".join(r.get("preview","") for r in top)

            metrics = {
                "lexical_f1": lexical_f1(ans, ref),
                "bow_cosine": bow_cosine(ans, ref),
                "context_recall": context_recall(ref, retrieved_text) if retrieved_text else 0.0,
            }
            rows.append({
                "question": q,
                "answer": ans,
                "reference": ref,
                "metrics": metrics,
            })
            console.print(f"[bold cyan]{q}[/bold cyan] -> F1={metrics['lexical_f1']:.3f} Cos={metrics['bow_cosine']:.3f} Ctx={metrics['context_recall']:.3f}")

    # aggregate
    avg = {
        "lexical_f1": mean(r["metrics"]["lexical_f1"] for r in rows) if rows else 0.0,
        "bow_cosine": mean(r["metrics"]["bow_cosine"] for r in rows) if rows else 0.0,
        "context_recall": mean(r["metrics"]["context_recall"] for r in rows) if rows else 0.0,
    }
    console.rule("[bold green]Averages")
    console.print(avg)

    # write a compact HTML report per run
    cfg_dict = cfg2.model_dump()
    summary = {
        "pipeline": pipe_id,
        "avg_metrics": avg,
        "num_examples": len(rows),
        "examples": rows[:20],  # cap inlined examples
    }
    report_path = write_simple_report(
        question=f"Benchmark: {pipe_id} on {Path(args.qa).name}",
        answer=json.dumps(summary, indent=2),
        cfg=cfg_dict,
        extras={"pipeline": pipe_id}
    )
    console.print(f"[green]Benchmark report written to {report_path}[/green]")

if __name__ == "__main__":
    main()

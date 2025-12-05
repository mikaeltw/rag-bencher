import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict

from rich.console import Console

from rag_bencher.config import load_config
from rag_bencher.eval.dataset_loader import load_texts_as_documents
from rag_bencher.eval.metrics import bow_cosine, context_recall, lexical_f1
from rag_bencher.eval.report import write_simple_report
from rag_bencher.pipelines.selector import PipelineSelection, select_pipeline

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a RAG pipeline on a QA set")
    ap.add_argument("--config", required=True)
    ap.add_argument("--qa", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    docs = load_texts_as_documents(cfg.data.paths)

    selection: PipelineSelection = select_pipeline(args.config, docs, cfg)
    chain = selection.chain
    debug = selection.debug
    pipe_id = selection.pipeline_id

    rows: list[Dict[str, float]] = []
    with open(args.qa, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            ref = ex["reference_answer"]
            ans = chain.invoke(q)
            dbg = debug()
            retrieved = ""
            if dbg.get("retrieved"):
                retrieved = "\n".join(r.get("preview", "") for r in dbg["retrieved"])
            elif dbg.get("candidates"):
                retrieved = "\n".join(r.get("preview", "") for r in dbg["candidates"][:5])
            metrics: Dict[str, float] = {
                "lexical_f1": lexical_f1(ans, ref),
                "bow_cosine": bow_cosine(ans, ref),
                "context_recall": context_recall(ref, retrieved) if retrieved else 0.0,
            }
            rows.append(metrics)
            console.print(
                f"[bold cyan]{q}[/bold cyan] -> F1={metrics['lexical_f1']:.3f} "
                f"Cos={metrics['bow_cosine']:.3f} "
                f"Ctx={metrics['context_recall']:.3f}"
            )
    avg: Dict[str, float] = {
        k: mean(r[k] for r in rows) if rows else 0.0 for k in ["lexical_f1", "bow_cosine", "context_recall"]
    }
    console.rule("[bold green]Averages")
    console.print(avg)
    summary: Dict[str, Any] = {"pipeline": pipe_id, "avg_metrics": avg, "num_examples": len(rows)}
    report_path = write_simple_report(
        question=f"Benchmark: {pipe_id} on {Path(args.qa).name}",
        answer=json.dumps(summary, indent=2),
        cfg=selection.config.model_dump(),
        extras={"pipeline": pipe_id},
    )
    console.print(f"[green]Benchmark report written to {report_path}[/green]")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()

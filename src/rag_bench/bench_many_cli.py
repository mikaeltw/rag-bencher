import argparse
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterator

from rich.console import Console

from rag_bencher.config import load_config
from rag_bencher.eval.dataset_loader import load_texts_as_documents
from rag_bencher.eval.metrics import bow_cosine, context_recall, lexical_f1
from rag_bencher.pipelines.selector import PipelineSelection, select_pipeline

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multiple configs and produce a combined HTML report")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--qa", required=True)
    args = ap.parse_args()

    first = sorted(glob.glob(args.configs))[0]
    cfg = load_config(first)
    docs = load_texts_as_documents(cfg.data.paths)

    def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    results: list[Dict[str, Any]] = []
    for p in sorted(glob.glob(args.configs)):
        selection: PipelineSelection = select_pipeline(p, docs)
        pid = selection.pipeline_id
        chain = selection.chain
        debug = selection.debug
        rows: list[Dict[str, float]] = []
        for ex in iter_jsonl(args.qa):
            q = ex["question"]
            ref = ex["reference_answer"]
            ans = chain.invoke(q)
            dbg = debug()
            retrieved = ""
            if dbg.get("retrieved"):
                retrieved = "\n".join(r.get("preview", "") for r in dbg["retrieved"])
            elif dbg.get("candidates"):
                retrieved = "\n".join(r.get("preview", "") for r in dbg["candidates"][:5])
            m = {
                "lexical_f1": lexical_f1(ans, ref),
                "bow_cosine": bow_cosine(ans, ref),
                "context_recall": context_recall(ref, retrieved) if retrieved else 0.0,
            }
            rows.append(m)
        avg = {k: mean(r[k] for r in rows) if rows else 0.0 for k in ["lexical_f1", "bow_cosine", "context_recall"]}
        console.print(f"[bold]{Path(p).name} ({pid})[/bold] -> {avg}")
        results.append({"config": Path(p).name, "pipeline": pid, **avg})

    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("reports") / f"summary-{ts}.html"
    out.parent.mkdir(exist_ok=True, parents=True)
    rows_html = "".join(
        (
            f"<tr>"
            f"<td>{r['config']}</td>"
            f"<td>{r['pipeline']}</td>"
            f"<td>{r['lexical_f1']:.3f}</td>"
            f"<td>{r['bow_cosine']:.3f}</td>"
            f"<td>{r['context_recall']:.3f}</td>"
            f"</tr>"
            for r in results
        )
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>rag-bencher multi-run</title>"
        f"<style>"
        f"body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
        f"max-width:1000px;margin:2rem auto;padding:0 1rem}}"
        f"table{{border-collapse:collapse;width:100%}}"
        f"th,td{{border:1px solid #ddd;padding:8px}}"
        f"</style></head><body>"
        f"<h1>rag-bencher multi-run summary</h1>"
        f"<table><thead><tr>"
        f"<th>Config</th><th>Pipeline</th><th>Lexical F1</th>"
        f"<th>BoW Cosine</th><th>Context Recall</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></body></html>"
    )
    out.write_text(html, encoding="utf-8")
    console.print(f"[green]Wrote {out}[/green]")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()

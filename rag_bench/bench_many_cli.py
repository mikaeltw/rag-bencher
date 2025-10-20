
import argparse, glob, json, os, yaml
from pathlib import Path
from statistics import mean
from rich.console import Console
from rag_bench.eval.dataset_loader import load_texts_as_documents
from rag_bench.eval.datasets import load_dataset
from rag_bench.config import load_config
from rag_bench.pipelines import naive_rag, multi_query as mq, rerank as rr, hyde as hy
from rag_bench.eval.metrics import lexical_f1, bow_cosine, context_recall

console = Console()

def choose(cfg_path: str, docs):
    cfg = load_config(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(os.path.expandvars(f.read())) or {}
    if "rerank" in raw:
        rrc = raw["rerank"]
        chain, debug = rr.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k,
                                      rerank_top_k=int(rrc.get("top_k",4)),
                                      method=str(rrc.get("method","auto")),
                                      cross_encoder_model=str(rrc.get("cross_encoder_model","BAAI/bge-reranker-base")))
        return "rerank", chain, debug, cfg
    if "multi_query" in raw:
        n = int(raw["multi_query"].get("n_queries",3))
        chain, debug = mq.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k, n_queries=n)
        return "multi_query", chain, debug, cfg
    if "hyde" in raw:
        chain, debug = hy.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k)
        return "hyde", chain, debug, cfg
    chain, debug = naive_rag.build_chain(docs, model=cfg.model.name, k=cfg.retriever.k)
    return "naive", chain, debug, cfg

def main():
    ap = argparse.ArgumentParser(description="Run multiple configs and produce a combined HTML report")
    ap.add_argument("--configs", required=True, help="Glob for config files (e.g., 'configs/*.yaml')")
    ap.add_argument("--qa", required=True, help="Path to JSONL QA set")
    ap.add_argument("--dataset", help="Dataset name (examples/datasets/...) to load docs from")
    args = ap.parse_args()

    # load docs
    if args.dataset:
        docs = load_dataset(args.dataset)
    else:
        # Use first config's data.paths as default
        first = sorted(glob.glob(args.configs))[0]
        cfg = load_config(first)
        from rag_bench.eval.dataset_loader import load_texts_as_documents
        docs = load_texts_as_documents(cfg.data.paths)

    # iterate configs
    import jsonlines
    # lightweight JSONL reader without external deps
    def iter_jsonl(path):
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    results = []
    for cfg_path in sorted(glob.glob(args.configs)):
        pipe_id, chain, debug, cfg = choose(cfg_path, docs)
        rows = []
        for ex in iter_jsonl(args.qa):
            q = ex["question"]; ref = ex["reference_answer"]
            ans = chain.invoke(q)
            dbg = debug()
            retrieved_text = ""
            if dbg.get("retrieved"):
                retrieved_text = "\n".join(r.get("preview","") for r in dbg["retrieved"])
            elif dbg.get("candidates"):
                retrieved_text = "\n".join(r.get("preview","") for r in dbg["candidates"][:5])
            metrics = {
                "lexical_f1": lexical_f1(ans, ref),
                "bow_cosine": bow_cosine(ans, ref),
                "context_recall": context_recall(ref, retrieved_text) if retrieved_text else 0.0,
            }
            rows.append(metrics)
        avg = {k: mean(r[k] for r in rows) if rows else 0.0 for k in ["lexical_f1","bow_cosine","context_recall"]}
        console.print(f"[bold]{Path(cfg_path).name} ({pipe_id})[/bold] -> {avg}")
        results.append({"config": Path(cfg_path).name, "pipeline": pipe_id, **avg})

    # write combined HTML
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("reports") / f"summary-{ts}.html"
    table_rows = ""
    for r in results:
        table_rows += f"<tr><td>{r['config']}</td><td>{r['pipeline']}</td><td>{r['lexical_f1']:.3f}</td><td>{r['bow_cosine']:.3f}</td><td>{r['context_recall']:.3f}</td></tr>"
    html = f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>rag-bench multi-run</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:8px}} th{{cursor:pointer}}
</style>
<script>
function sortTable(n){{
  var table=document.getElementById("res"); var switching=true; var dir="asc"; var switchcount=0;
  while(switching){{ switching=false; var rows=table.rows; for(var i=1;i<rows.length-1;i++){{ var x=rows[i].getElementsByTagName("TD")[n]; var y=rows[i+1].getElementsByTagName("TD")[n];
      var cmp = isNaN(parseFloat(x.innerHTML)) ? x.innerHTML.localeCompare(y.innerHTML) : (parseFloat(x.innerHTML)-parseFloat(y.innerHTML));
      if((dir=="asc" && cmp>0) || (dir=="desc" && cmp<0)){{ rows[i].parentNode.insertBefore(rows[i+1], rows[i]); switching=true; switchcount++; break; }}
  }} if(!switching && switchcount==0){{ dir=(dir=="asc")?"desc":"asc"; switching=true; }} }}
}}
</script>
</head><body>
<h1>rag-bench multi-run summary</h1>
<table id="res"><thead><tr>
<th onclick="sortTable(0)">Config</th><th onclick="sortTable(1)">Pipeline</th>
<th onclick="sortTable(2)">Lexical F1</th><th onclick="sortTable(3)">BoW Cosine</th><th onclick="sortTable(4)">Context Recall</th>
</tr></thead><tbody>
{table_rows}
</tbody></table>
</body></html>
"""
    out.write_text(html, encoding="utf-8")
    console.print(f"[green]Wrote {out}[/green]")

if __name__ == "__main__":
    main()

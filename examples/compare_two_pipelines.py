"""Compare two rag-bencher configs over a small QA set.

Run directly:
    uv run python examples/compare_two_pipelines.py
or adjust `CONFIGS` to point at your own YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from rag_bencher.config import load_config
from rag_bencher.eval.dataset_loader import load_texts_as_documents
from rag_bencher.eval.metrics import bow_cosine, context_recall, lexical_f1
from rag_bencher.pipelines.selector import select_pipeline

CONFIGS = ("configs/wiki.yaml", "configs/rerank.yaml")
QA_EXAMPLES: List[Dict[str, str]] = [
    {"question": "What is LangChain?", "reference_answer": "A framework for building LLM-powered apps."},
    {
        "question": "What is retrieval-augmented generation?",
        "reference_answer": "It retrieves documents and feeds them to a generator for grounded answers.",
    },
]
DOCS = load_texts_as_documents(["examples/data/sample.txt"])


def evaluate(config_path: str) -> Tuple[str, Dict[str, float]]:
    cfg = load_config(config_path)
    selection = select_pipeline(config_path, DOCS, cfg)
    chain = selection.chain
    debug = selection.debug

    scores: List[Dict[str, float]] = []
    for qa in QA_EXAMPLES:
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


def main() -> None:
    for cfg in CONFIGS:
        pid, metrics = evaluate(cfg)
        print(f"{Path(cfg).name} ({pid}) -> {metrics}")


if __name__ == "__main__":
    main()

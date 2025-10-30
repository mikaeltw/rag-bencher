from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import yaml
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable

from rag_bench.config import BenchConfig, load_config
from rag_bench.pipelines import hyde as hy
from rag_bench.pipelines import multi_query as mq
from rag_bench.pipelines import naive_rag
from rag_bench.pipelines import rerank as rr
from rag_bench.providers.base import build_chat_adapter, build_embeddings_adapter


@dataclass(frozen=True)
class PipelineSelection:
    """Container describing a configured pipeline."""

    pipeline_id: str
    config: BenchConfig
    chain: RunnableSerializable[str, str]
    debug: Callable[[], Mapping[str, Any]]


def _load_raw_config(cfg_path: str) -> dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(os.path.expandvars(f.read())) or {}


def _build_provider_adapters(cfg: BenchConfig) -> tuple[Optional[RunnableSerializable[Any, Any]], Optional[Any]]:
    provider_cfg = cfg.model_dump().get("provider")
    chat_adapter = build_chat_adapter(provider_cfg) if getattr(cfg, "provider", None) else None
    emb_adapter = build_embeddings_adapter(provider_cfg) if getattr(cfg, "provider", None) else None
    llm_obj = chat_adapter.to_langchain() if chat_adapter else None
    emb_obj = emb_adapter.to_langchain() if emb_adapter else None
    return llm_obj, emb_obj


def select_pipeline(
    cfg_path: str,
    docs: list[Document],
    cfg: BenchConfig | None = None,
) -> PipelineSelection:
    """Build the runnable chain and debug hook for the pipeline described by ``cfg_path``.

    Parameters
    ----------
    cfg_path:
        Path to the YAML configuration file.
    docs:
        Corpus documents the pipeline will index/retrieve from.
    cfg:
        Optional pre-loaded BenchConfig to avoid re-parsing.
    """
    bench_cfg = cfg or load_config(cfg_path)
    llm_obj, emb_obj = _build_provider_adapters(bench_cfg)
    raw_cfg = _load_raw_config(cfg_path)

    if "rerank" in raw_cfg:
        rrc = raw_cfg["rerank"] or {}
        rerank_top_k = int(rrc.get("top_k", 4))
        method = str(rrc.get("method", "cosine"))
        cross_encoder_model = str(rrc.get("cross_encoder_model", "BAAI/bge-reranker-base"))
        chain, debug = rr.build_chain(
            docs,
            model=bench_cfg.model.name,
            k=bench_cfg.retriever.k,
            rerank_top_k=rerank_top_k,
            method=method,
            cross_encoder_model=cross_encoder_model,
            llm=llm_obj,
            embeddings=emb_obj,
        )
        pipeline_id = "rerank"
    elif "multi_query" in raw_cfg:
        mq_cfg = raw_cfg["multi_query"] or {}
        n_queries = int(mq_cfg.get("n_queries", 3))
        chain, debug = mq.build_chain(
            docs,
            model=bench_cfg.model.name,
            k=bench_cfg.retriever.k,
            n_queries=n_queries,
            llm=llm_obj,
            embeddings=emb_obj,
        )
        pipeline_id = "multi_query"
    elif "hyde" in raw_cfg:
        chain, debug = hy.build_chain(
            docs,
            model=bench_cfg.model.name,
            k=bench_cfg.retriever.k,
            llm=llm_obj,
            embeddings=emb_obj,
        )
        pipeline_id = "hyde"
    else:
        chain, debug = naive_rag.build_chain(
            docs,
            model=bench_cfg.model.name,
            k=bench_cfg.retriever.k,
            llm=llm_obj,
            embeddings=emb_obj,
        )
        pipeline_id = "naive"

    return PipelineSelection(pipeline_id=pipeline_id, config=bench_cfg, chain=chain, debug=debug)

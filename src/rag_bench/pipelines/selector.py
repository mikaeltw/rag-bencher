from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable

from rag_bencher.config import BenchConfig, load_config
from rag_bencher.pipelines import hyde as hy
from rag_bencher.pipelines import multi_query as mq
from rag_bencher.pipelines import naive_rag
from rag_bencher.pipelines import rerank as rr
from rag_bencher.providers.base import build_chat_adapter, build_embeddings_adapter


@dataclass(frozen=True)
class PipelineSelection:
    """Container describing a configured pipeline."""

    pipeline_id: str
    config: BenchConfig
    chain: RunnableSerializable[str, str]
    debug: Callable[[], Mapping[str, Any]]


def _build_provider_adapters(cfg: BenchConfig) -> tuple[Optional[RunnableSerializable[Any, Any]], Optional[Any]]:
    provider_obj = getattr(cfg, "provider", None)
    provider_cfg = provider_obj.model_dump() if provider_obj else None
    chat_adapter = build_chat_adapter(provider_cfg) if provider_cfg else None
    emb_adapter = build_embeddings_adapter(provider_cfg) if provider_cfg else None
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

    if bench_cfg.rerank is not None:
        rrc = bench_cfg.rerank
        chain, debug = rr.build_chain(
            docs,
            model=bench_cfg.model.name,
            k=bench_cfg.retriever.k,
            rerank_top_k=rrc.top_k,
            method=rrc.method,
            cross_encoder_model=rrc.cross_encoder_model or "BAAI/bge-reranker-base",
            llm=llm_obj,
            embeddings=emb_obj,
        )
        pipeline_id = "rerank"
    elif bench_cfg.multi_query is not None:
        mq_cfg = bench_cfg.multi_query
        chain, debug = mq.build_chain(
            docs,
            model=bench_cfg.model.name,
            k=bench_cfg.retriever.k,
            n_queries=mq_cfg.n_queries,
            llm=llm_obj,
            embeddings=emb_obj,
        )
        pipeline_id = "multi_query"
    elif bench_cfg.hyde is not None:
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

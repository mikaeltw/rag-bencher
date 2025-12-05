import argparse
import os
from typing import Any, Optional, cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSerializable
from rich.console import Console

from rag_bencher.config import BenchConfig, load_config
from rag_bencher.eval.dataset_loader import load_texts_as_documents
from rag_bencher.pipelines import naive_rag
from rag_bencher.providers.base import build_chat_adapter, build_embeddings_adapter
from rag_bencher.utils.cache import cache_get, cache_set
from rag_bencher.utils.callbacks.usage import UsageTracker
from rag_bencher.utils.repro import set_seeds
from rag_bencher.vector.base import VectorBackend, build_vector_backend

console = Console()


def _pick_llm(cfg: BenchConfig) -> RunnableSerializable[Any, Any]:
    """Return a LangChain LLM object based on offline flag."""
    if getattr(cfg.runtime, "offline", False):
        # Local, CPU-friendly text2text model that follows instructions better than GPT-2.
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from transformers import pipeline as hf_pipeline

        model_id = os.getenv("RAG_BENCH_OFFLINE_MODEL", "google/flan-t5-small")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        gen = hf_pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tok,
            device=-1,
            max_new_tokens=160,
        )

        generation_config = getattr(gen.model, "generation_config", None)
        if generation_config is not None:
            generation_config.update(
                max_new_tokens=160,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        llm: RunnableSerializable[Any, Any] = HuggingFacePipeline(pipeline=gen)
        # Encourage clean stops if the template tries to ask follow-up questions.
        return cast(RunnableSerializable[Any, Any], llm.bind(stop=["\nQuestion:", "###END"]))
    else:
        # Cloud (OpenAI via langchain-openai)
        prov = getattr(cfg, "provider", None)
        if prov:
            chat = build_chat_adapter(cfg.model_dump().get("provider"))
            if not chat:
                raise RuntimeError("Provider configured but chat adapter unavailable")
            return cast(RunnableSerializable[Any, Any], chat.to_langchain())
        else:
            from langchain_openai import ChatOpenAI

            return cast(RunnableSerializable[Any, Any], ChatOpenAI(model=cfg.model.name, temperature=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--question", required=True)
    args = ap.parse_args()

    set_seeds(42)

    cfg = load_config(args.config)

    # Apply device preference early so torch/embeddings respect it
    dev = getattr(cfg.runtime, "device", "auto")
    if dev == "cpu":
        os.environ.setdefault("RAG_BENCH_DEVICE", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    elif dev in ("cuda",):
        os.environ.setdefault("RAG_BENCH_DEVICE", "cuda")

    docs: list[Document] = load_texts_as_documents(cfg.data.paths)

    # Embeddings: if you’re using the factory, it respects CPU/GPU globally.
    emb: Optional[Embeddings] = None
    prov = getattr(cfg, "provider", None)
    if prov:
        adapter = build_embeddings_adapter(cfg.model_dump().get("provider"))
        if adapter:
            emb = adapter.to_langchain()

    # Vector retriever (optional; safe fallback)
    vec: Optional[VectorBackend] = build_vector_backend(cfg.model_dump().get("vector"))
    retr: Optional[BaseRetriever] = None
    if vec and emb is not None:
        try:
            retr = vec.make_retriever(docs=None, embeddings=emb, k=cfg.retriever.k)
        except Exception:
            retr = None

    # Select LLM by offline flag
    llm_obj = _pick_llm(cfg)

    chain, _meta = naive_rag.build_chain(
        docs, model=cfg.model.name, k=cfg.retriever.k, llm=llm_obj, embeddings=emb, retriever=retr
    )

    prompt = args.question
    cached = cache_get(cfg.model.name, prompt)
    if cached is None:
        ans = chain.invoke(prompt, config={"callbacks": [UsageTracker()]})
        cache_set(cfg.model.name, prompt, ans)
    else:
        ans = cached

    console.print(ans)


if __name__ == "__main__":  # pragma: no cover - exercised via CLI entrypoint
    main()

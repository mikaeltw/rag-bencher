from typing import Any, Dict, List, Optional, cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSerializable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy.typing import ArrayLike

from rag_bencher.pipelines.base import BuildResult
from rag_bencher.pipelines.utils import resolve_chat_llm
from rag_bencher.utils.factories import make_hf_embeddings
from rag_bencher.vector.local import build_local_vectorstore


def _cosine(u: ArrayLike, v: ArrayLike) -> float:
    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)
    if un == 0 or vn == 0:
        return 0.0
    return float(np.dot(u, v) / (un * vn))


def build_chain(
    docs: List[Document],
    model: str = "gpt-4o-mini",
    k: int = 8,
    rerank_top_k: int = 4,
    method: str = "cosine",
    cross_encoder_model: str = "BAAI/bge-reranker-base",
    llm: Optional[RunnableSerializable[Any, Any]] = None,
    embeddings: Optional[Embeddings] = None,
) -> BuildResult:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = embeddings or make_hf_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = build_local_vectorstore(splits, embed)

    class _ContextBuilder:
        def __init__(self) -> None:
            self._last_debug: Dict[str, Any] = {"pipeline": "rerank", "method": method, "candidates": []}

        def __call__(self, question: str) -> str:
            candidates = vect.similarity_search(question, k=k)
            qv = embed.embed_query(question)
            scores: List[tuple[Document, float]] = []
            for d in candidates:
                dv = embed.embed_query(d.page_content)
                scores.append((d, _cosine(qv, dv)))
            scores.sort(key=lambda x: x[1], reverse=True)
            chosen = [d for d, _ in scores[:rerank_top_k]]
            context = "\n\n".join(d.page_content for d in chosen)
            self._last_debug = {
                "pipeline": "rerank",
                "method": method,
                "rerank_top_k": rerank_top_k,
                "candidates": [
                    {
                        "score": float(sc),
                        "preview": doc.page_content[:160],
                        "source": doc.metadata.get("source", ""),
                    }
                    for doc, sc in scores[:20]
                ],
            }
            return context

        @property
        def last_debug(self) -> Dict[str, Any]:
            return self._last_debug

    context_builder = _ContextBuilder()
    template = (
        "You are a helpful assistant. Use the context to answer.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = PromptTemplate.from_template(template)
    llm_answer = resolve_chat_llm(model, override=llm)

    chain = cast(
        RunnableSerializable[str, str],
        {
            "context": RunnableLambda(context_builder),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm_answer
        | StrOutputParser(),
    )

    def debug() -> Dict[str, Any]:
        return context_builder.last_debug

    return chain, debug

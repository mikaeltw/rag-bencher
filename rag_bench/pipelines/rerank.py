
from __future__ import annotations
from typing import List, Tuple
import math
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document

@dataclass
class RerankConfig:
    method: str = "auto"     # "auto" | "cross_encoder" | "cosine"
    top_k: int = 4
    cross_encoder_model: str = "BAAI/bge-reranker-base"

def _try_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception:
        return None

def _cosine(u, v):
    import numpy as np
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
    method: str = "auto",
    cross_encoder_model: str = "BAAI/bge-reranker-base",
):
    # 1) Build base vector index
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = FAISS.from_documents(splits, embed)

    # 2) Prepare reranker
    ce = None
    use_cosine = False
    if method in ("auto", "cross_encoder"):
        ce = _try_cross_encoder(cross_encoder_model)
        if ce is None and method == "cross_encoder":
            # explicit cross_encoder requested but unavailable → fallback to cosine
            use_cosine = True
        elif ce is None and method == "auto":
            use_cosine = True
    else:
        use_cosine = True

    # 3) Define retrieval + rerank function
    def build_context(question: str):
        # Get a larger candidate pool (k) then rerank down to rerank_top_k
        candidates = vect.similarity_search(question, k=k)

        scores = []
        if ce is not None:
            pairs = [(question, d.page_content) for d in candidates]
            # CE returns higher = better
            s = ce.predict(pairs).tolist()
            scores = list(zip(candidates, s))
        else:
            # cosine on embedding space as a light fallback
            qv = embed.embed_query(question)
            dv = [embed.embed_query(d.page_content) for d in candidates]
            s = [_cosine(qv, vec) for vec in dv]
            scores = list(zip(candidates, s))

        # sort by score desc
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = [d for d, _ in scores[:rerank_top_k]]
        context = "\n\n".join(d.page_content for d in chosen)

        build_context._last_debug = {
            "pipeline": "rerank",
            "method": "cross_encoder" if ce is not None and not use_cosine else "cosine",
            "rerank_top_k": rerank_top_k,
            "candidates": [
                {"score": float(sc), "preview": doc.page_content[:160], "source": doc.metadata.get("source","")}
                for doc, sc in scores[:20]
            ],
        }
        return context

    build_context._last_debug = {
        "pipeline": "rerank",
        "method": "unknown",
        "rerank_top_k": rerank_top_k,
        "candidates": [],
    }

    # 4) Answer LLM
    template = (
        "You are a helpful assistant. Use the context to answer.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model=model, temperature=0)

    chain = (
        {"context": RunnableLambda(build_context), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    def debug():
        return getattr(build_context, "_last_debug", {"pipeline": "rerank"})

    return chain, debug

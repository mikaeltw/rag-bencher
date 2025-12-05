from typing import Any, Callable, Dict, List, Optional, cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_bencher.pipelines.base import BuildResult
from rag_bencher.pipelines.utils import has_openai_key, resolve_chat_llm
from rag_bencher.utils.factories import make_hf_embeddings
from rag_bencher.vector.local import build_local_vectorstore

GEN_PROMPT = """You are an expert at generating diverse search queries.
Produce {n} different queries that could retrieve context to answer the user's question.
Return one query per line, no numbering.

Question: {question}
"""


def _fallback_queries(question: str, n: int) -> List[str]:
    variants = [
        question,
        f"Background for: {question}",
        f"Key facts related to: {question}",
        f"Overview: {question}",
        f"ELI5: {question}",
    ]
    return variants[: max(1, n)]


def _dedupe_queries(base: str, generated: List[str], limit: int) -> List[str]:
    uniq: List[str] = []
    for candidate in [base] + generated:
        if candidate not in uniq:
            uniq.append(candidate)
        if len(uniq) >= max(1, limit):
            break
    return uniq


def build_chain(
    docs: List[Document],
    model: str = "gpt-4o-mini",
    k: int = 4,
    n_queries: int = 3,
    llm: Optional[RunnableSerializable[Any, Any]] = None,
    embeddings: Optional[Embeddings] = None,
) -> BuildResult:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = embeddings or make_hf_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = build_local_vectorstore(splits, embed)

    llm_answer = resolve_chat_llm(model, override=llm)
    openai_ok = has_openai_key()
    if openai_ok and llm is None:
        llm_gen = ChatOpenAI(model=model, temperature=0)
        gen_tmpl = PromptTemplate.from_template(GEN_PROMPT)

        def gen_queries(q: str) -> List[str]:
            text = (gen_tmpl | llm_gen | StrOutputParser()).invoke({"n": n_queries, "question": q})
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return _dedupe_queries(q, lines, n_queries)

    else:

        def gen_queries(q: str) -> List[str]:
            return _fallback_queries(q, n_queries)

    class _ContextBuilder:
        def __init__(self, query_fn: Callable[[str], List[str]]) -> None:
            self._query_fn = query_fn
            self._last_debug: Dict[str, Any] = {"pipeline": "multi_query", "queries": [], "retrieved": []}

        def __call__(self, question: str) -> str:
            queries = self._query_fn(question)
            seen: set[str] = set()
            aggregated: List[Document] = []
            for qr in queries:
                docs_q = vect.similarity_search(qr, k=k)
                for d in docs_q:
                    key = d.page_content[:200]
                    if key not in seen:
                        seen.add(key)
                        aggregated.append(d)
            context = "\n\n".join(d.page_content for d in aggregated[: max(k, len(aggregated))])
            self._last_debug = {
                "pipeline": "multi_query",
                "queries": queries,
                "retrieved": [
                    {"source": d.metadata.get("source", ""), "preview": d.page_content[:160]} for d in aggregated
                ],
            }
            return context

        @property
        def last_debug(self) -> Dict[str, Any]:
            return self._last_debug

    context_builder = _ContextBuilder(gen_queries)

    template = (
        "You are a helpful assistant. Use the context to answer.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = PromptTemplate.from_template(template)

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

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

HYP_PROMPT = """You will draft a hypothetical answer to help retrieve relevant passages.
Question: {question}
Draft a concise, factual paragraph:"""


def _fallback_hypothesis(question: str) -> str:
    return f"This is a draft answer about: {question}. It outlines likely definitions, key concepts, and use cases."


def build_chain(
    docs: List[Document],
    model: str = "gpt-4o-mini",
    k: int = 4,
    llm: Optional[RunnableSerializable[Any, Any]] = None,
    embeddings: Optional[Embeddings] = None,
) -> BuildResult:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = embeddings or make_hf_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = build_local_vectorstore(splits, embed)

    openai_ok = has_openai_key()
    if openai_ok and llm is None:
        llm_h = ChatOpenAI(model=model, temperature=0)
        hyp_tmpl = PromptTemplate.from_template(HYP_PROMPT)

        def gen_hyp(q: str) -> str:
            return (hyp_tmpl | llm_h | StrOutputParser()).invoke({"question": q}).strip()

    else:

        def gen_hyp(q: str) -> str:
            return _fallback_hypothesis(q)

    llm_answer = resolve_chat_llm(model, override=llm)

    class _ContextBuilder:
        def __init__(self, generator: Callable[[str], str]) -> None:
            self._generator = generator
            self._last_debug: Dict[str, Any] = {"pipeline": "hyde", "hypothesis": "", "retrieved": []}

        def __call__(self, question: str) -> str:
            hyp = self._generator(question)
            docs_h = vect.similarity_search(hyp, k=k)
            context = "\n\n".join(d.page_content for d in docs_h)
            self._last_debug = {
                "pipeline": "hyde",
                "hypothesis": hyp,
                "retrieved": [
                    {"source": d.metadata.get("source", ""), "preview": d.page_content[:160]} for d in docs_h
                ],
            }
            return context

        @property
        def last_debug(self) -> Dict[str, Any]:
            return self._last_debug

    context_builder = _ContextBuilder(gen_hyp)

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


from __future__ import annotations
from typing import List
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document

HYP_PROMPT = """You will draft a hypothetical answer to help retrieve relevant passages.
Question: {question}
Draft a concise, factual paragraph:"""

def _fallback_hypothesis(question: str) -> str:
    return f"This is a draft answer about: {question}. It outlines likely definitions, key concepts, and common use cases."

def build_chain(docs: List[Document], model: str = "gpt-4o-mini", k: int = 4):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = FAISS.from_documents(splits, embed)

    openai_ok = bool(os.environ.get("OPENAI_API_KEY"))
    if openai_ok:
        llm_h = ChatOpenAI(model=model, temperature=0)
        hyp_tmpl = PromptTemplate.from_template(HYP_PROMPT)
        def gen_hyp(q: str) -> str:
            return (hyp_tmpl | llm_h | StrOutputParser()).invoke({"question": q}).strip()
    else:
        def gen_hyp(q: str) -> str:
            return _fallback_hypothesis(q)

    llm_answer = ChatOpenAI(model=model, temperature=0)

    def build_context(question: str) -> str:
        hyp = gen_hyp(question)
        docs_h = vect.similarity_search(hyp, k=k)
        context = "\n\n".join(d.page_content for d in docs_h)
        build_context._last_debug = {
            "pipeline": "hyde",
            "hypothesis": hyp,
            "retrieved": [ {"source": d.metadata.get("source",""), "preview": d.page_content[:160]} for d in docs_h ]
        }
        return context

    build_context._last_debug = {"pipeline": "hyde", "hypothesis": "", "retrieved": []}

    template = (
        "You are a helpful assistant. Use the context to answer.
"
        "If the answer is not in the context, say you don't know.
"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": RunnableLambda(build_context), "question": RunnablePassthrough()}
        | prompt
        | llm_answer
        | StrOutputParser()
    )

    def debug():
        return getattr(build_context, "_last_debug", {"pipeline": "hyde"})

    return chain, debug

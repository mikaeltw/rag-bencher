
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document

def build_chain(docs: List[Document], model: str = "gpt-4o-mini", k: int = 4):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = FAISS.from_documents(splits, embed)
    retriever = vect.as_retriever(search_kwargs={"k": k})

    template = (
        "You are a helpful assistant. Use the context to answer.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model=model, temperature=0)

    def context_joiner(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        { "context": retriever | context_joiner,
          "question": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
    )

    def debug():
        return {"pipeline": "naive_rag"}

    return chain, debug

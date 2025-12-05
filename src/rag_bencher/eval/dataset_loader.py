from pathlib import Path

from langchain_core.documents import Document


def load_texts_as_documents(paths: list[str]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        docs.append(Document(page_content=Path(p).read_text(encoding="utf-8"), metadata={"source": p}))
    return docs

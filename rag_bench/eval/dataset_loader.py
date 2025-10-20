
from typing import List
from pathlib import Path
from langchain.docstore.document import Document

def load_texts_as_documents(paths: List[str]) -> List[Document]:
    docs = []
    for p in paths:
        txt = Path(p).read_text(encoding="utf-8")
        docs.append(Document(page_content=txt, metadata={"source": str(p)}))
    return docs

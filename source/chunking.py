from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    spillter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_documents = []

    for document in documents:
        text = spillter.split_text(document.page_content)
        for i, t in enumerate(text, start=1):
            metadata = dict(document.metadata) if document.metadata else {}
            metadata["chunk_index"] = i
            document = Document(page_content=t, metadata=metadata)
            splitted_documents.append(document)

    return splitted_documents
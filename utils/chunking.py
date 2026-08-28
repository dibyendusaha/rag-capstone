from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
    spillter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_documents = []

    for chunk, document in enumerate(documents, start=1):
        text = spillter.split_text(document)
        metadata = dict(document.metadata) if document.metadata else {}
        metadata.chunk_index = chunk
        document = Document(page_content=text, metadata=metadata)
        splitted_documents.append(document)

    return splitted_documents
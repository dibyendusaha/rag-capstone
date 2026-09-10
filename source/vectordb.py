import os
from typing import Any, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

def sync_documents_to_vector_db(collection_name: str, embedding_engine: Any, documents: List[Document]) -> Chroma:
    persist_path = os.path.join(os.path.dirname(os.getcwd()), "store", "chroma_db")

    if os.path.exists(persist_path) and os.listdir(persist_path):
        vectordb = Chroma(
            collection_name=f"capstone_rag_{collection_name}",
            persist_directory=persist_path,
            embedding_function=embedding_engine,
        )

        if documents:
            vectordb.add_documents(documents=documents)
            print(f"✅ Successfully appended {len(documents)} new chunks to the existing database.")

        else:
            print("⚪ No new documents provided — loaded existing index only.")

    else:
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding_engine,
            collection_name=f"capstone_rag_{collection_name}",
            persist_directory=persist_path,
        )
        print(f"🚀 Successfully created new database and indexed {len(documents)} chunks.")

    return vectordb
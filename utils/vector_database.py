import os
from typing import Any, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

def sync_documents_to_vector_db(embedding_engine: Any, documents: List[Document]):
    persist_path = os.path.join(os.path.dirname(os.getcwd()), "store", "chroma_db")

    if os.path.exists(persist_path) and os.listdir(persist_path):
        vector_database = Chroma(
            collection_name="capstone_rag",
            persist_directory=persist_path,
            embedding_function=embedding_engine,
        )

        if documents:
            vector_database.add_documents(documents=documents)
            print(f"✅ Successfully appended {len(documents)} new chunks to the existing database.")

        else:
            print("⚪ No new documents provided — loaded existing index only.")

    else:
        vector_database = Chroma.from_documents(
            documents=documents,
            embedding=embedding_engine,
            collection_name="capstone_rag",
            persist_directory=persist_path,
        )
        print(f"🚀 Successfully created new database and indexed {len(documents)} chunks.")

    return vector_database
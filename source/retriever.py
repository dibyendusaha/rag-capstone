from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

def get_retriever(provider_name: str, vectordb: Chroma) -> MultiQueryRetriever:
    provider = provider_name.lower().strip()

    if provider == "openai":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    elif provider == "google" or provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        
    elif provider == "nvidia":
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
        
    else:
        raise ValueError(f"Provider '{provider_name}' is unsupported.")

    base_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    return multi_query_retriever
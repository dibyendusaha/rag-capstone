from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings_provider(provider_name: str) -> GoogleGenerativeAIEmbeddings | OpenAIEmbeddings | HuggingFaceEmbeddings:
    provider = provider_name.lower().strip()

    if provider == "google" or provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

    elif provider == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    elif provider == "nvidia":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    else:
        raise ValueError(
            f"Unsupported provider: '{provider_name}'. "
            f"Please choose from 'google/gemini', 'openai', or 'nvidia'."
        )
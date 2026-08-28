import os
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

def get_embeddings_provider(provider_name: str) -> Any:
    provider = provider_name.lower().strip()

    if provider == "google" or provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

    elif provider == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    elif provider == "nvidia":
        return NVIDIAEmbeddings(
            model="NV-Embed-QA"
        )

    else:
        raise ValueError(
            f"Unsupported provider: '{provider_name}'. "
            f"Please choose from 'google/gemini', 'openai', or 'nvidia'."
        )
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

def build_ag_chain(provider_name: str):
    provider = provider_name.lower().strip()
        
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    elif provider == "google" or provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        
    elif provider == "nvidia":
        return ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct", temperature=0)
        
    else:
        raise ValueError(f"Provider '{provider_name}' is unsupported.")


    system_prompt = ""

    prompt = ""
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from memory import get_active_context, get_memory_summary

def build_rag_chain(provider_name: str, retriever: MultiQueryRetriever):
    provider = provider_name.lower().strip()
        
    if provider == "openai":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    elif provider == "google" or provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        
    elif provider == "nvidia":
        llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct", temperature=0)
        
    else:
        raise ValueError(f"Provider '{provider_name}' is unsupported.")


    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)


    system_prompt = """
    You are a helpful AI assistant.
    You have access to two types of memory:
    1️⃣ A summary of past conversations.
    2️⃣ The most recent question-answer turns for short-term context.
    Always use both to maintain context and continuity naturally.
    """

    prompt = PromptTemplate.from_template("""
    System Prompt:
    {system_prompt}

    Conversation Summary Memory:
    {memory_summary}

    Recent Chat History (last few turns):
    {active_memory}

    Retrieved Context:
    {context}

    User Question:
    {question}

    If the user asks to summarize or refer to earlier parts of the conversation,
    rely primarily on the chat history and memory summary.
    Otherwise, combine retrieved context and memories to answer effectively.
    """)

    rag_chain = RunnableParallel({
        "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x["question"]),
        "system_prompt": RunnableLambda(lambda _: system_prompt),
        "active_memory": RunnableLambda(lambda x, config: get_active_context(config["configurable"]["session_id"])),
        "memory_summary": RunnableLambda(lambda x, config: get_memory_summary(config["configurable"]["session_id"], config["configurable"]["provider"]))
    }) | prompt | llm

    return rag_chain
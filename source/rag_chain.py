from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# from memory import get_active_context, get_memory_summary
try:
    from source.native_memory import get_session_summary_memory, get_formatted_session_active_memory
except ModuleNotFoundError:
    from native_memory import get_session_summary_memory, get_formatted_session_active_memory

def build_rag_chain(provider_name: str, retriever: MultiQueryRetriever, config = None):
    provider = provider_name.lower().strip()
        
    if provider == "openai":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    elif provider == "google" or provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        
    elif provider == "groq":
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
        
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

    """
    If you are using memory.py
    then use the below rag_chain RunnableSerializable
    """
    """
        rag_chain = RunnableParallel({
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda x: x["question"]),
            "system_prompt": RunnableLambda(lambda _: system_prompt),
            "active_memory": RunnableLambda(lambda x, config: get_active_context(config["configurable"]["session_id"])),
            "memory_summary": RunnableLambda(lambda x, config: get_memory_summary(config["configurable"]["session_id"], config["configurable"]["provider"]))
        }) | prompt | llm
    """

    """
    If you are using native_memory.py
    teh use the below rag_chain RunnableSerializable
    """
    rag_chain = RunnableParallel({
        "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x["question"]),
        "system_prompt": RunnableLambda(lambda _: system_prompt),
        "active_memory": RunnableLambda(lambda x, config: x["current_memory"] if "current_memory" in x else get_formatted_session_active_memory(config["configurable"]["session_id"])),
        "memory_summary": RunnableLambda(lambda x, config: x["memory_summary"] if "memory_summary" in x else get_session_summary_memory(config["configurable"]["session_id"]))
    }) | prompt | llm

    return rag_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable, RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA


def get_llm_provider(provider_name: str):
    provider = provider_name.lower().strip()
    
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    elif provider == "google" or provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        
    elif provider == "nvidia":
        return ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct", temperature=0)
        
    else:
        raise ValueError(f"Provider '{provider_name}' is unsupported.")


SESSION_MEMORY = {}
SESSION_ACTIVE_MEMORY = {}
SESSION_SUMMARY_MEMORY = {}

def get_session_memory_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = InMemoryChatMessageHistory()

    return SESSION_MEMORY[session_id]


def get_session_active_memory(session_id: str) -> list:
    return SESSION_ACTIVE_MEMORY.get(session_id, [])


def update_session_active_memory(session_id: str, user_input: str, ai_output: str) -> list[dict[str, str]]:
    if session_id not in SESSION_ACTIVE_MEMORY:
        SESSION_ACTIVE_MEMORY[session_id] = []
    SESSION_ACTIVE_MEMORY[session_id].append({"human": user_input, "ai": ai_output})


def get_formatted_session_active_memory(session_id) -> str:
    history = SESSION_ACTIVE_MEMORY[session_id]

    if not history:
        return "No history found for this chat."

    return "\n\n".join(f"You: {chat["human"]}\nAI: {chat["ai"]}" for chat in history)


def get_session_summary_memory(session_id: str) -> str:
    return SESSION_SUMMARY_MEMORY.get(session_id, "No summary found for this chat.")


def summarize_chat_prompt_chain(llm) -> RunnableSerializable:
    summary_prompt = PromptTemplate(
        template="""
        You are an expert summarizer. Extend the previous summary by incorporating the new messages below chronologically to keep a concise running summary of the conversation.
        Current Summary:
        {existing_summary}

        New Messages:
        {new_messages}

        Updated Summary:
        """,
        input_variables=["existing_summary", "new_messages"],
    )
    return summary_prompt | llm


def add_memory_to_rag_chain(session_id: str, rag_chain: RunnableSerializable, enabled: bool, provider: str):
    if not enabled:
        print(f"🧠 Memory is not enabled for this chat.")
        return rag_chain

    llm = get_llm_provider(provider_name=provider)
    summary_chain = summarize_chat_prompt_chain(llm)

    def process_with_native_memory(input: dict, config = None):
        user_input = input["question"]
        
        history_obj = get_session_memory_history(session_id=session_id)
        current_messages = history_obj.messages

        rag_payload = {
            "question": user_input,
            "current_memory": current_messages,
            "memory_summary": get_session_summary_memory(session_id=session_id),
        }
        response = rag_chain.invoke(rag_payload)
        ai_output = getattr(response, "content", str(response))

        history_obj.add_user_message(message=user_input)
        history_obj.add_ai_message(message=ai_output)
        update_session_active_memory(session_id=session_id, user_input=user_input, ai_output=ai_output)

        existing_summary = get_session_summary_memory(session_id=session_id)
        new_message = f"You: {user_input}\nAI: {ai_output}"
        summary_response = summary_chain.invoke({
            "existing_summary": existing_summary,
            "new_messages": new_message
        })
        SESSION_SUMMARY_MEMORY[session_id] = getattr(summary_response, "content", str(summary_response))

        return response

    return RunnableLambda(process_with_native_memory)
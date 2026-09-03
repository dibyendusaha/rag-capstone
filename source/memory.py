from langchain_classic.schema.runnable import RunnableLambda
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

SESSION_MESSAGE_HISTORY = {}
SESSION_MESSAGE_SUMMARY = {}
SESSION_ACTIVE_MEMORY = {}

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

def get_session_message_history(session_id: str):
    if session_id not in SESSION_MESSAGE_HISTORY:
        SESSION_MESSAGE_HISTORY[session_id] = InMemoryChatMessageHistory()

    return SESSION_MESSAGE_HISTORY[session_id]

def get_session_message_summary(session_id: str, provider_name: str) -> ConversationSummaryMemory:
    if session_id not in SESSION_MESSAGE_SUMMARY:
        SESSION_MESSAGE_SUMMARY[session_id] = ConversationSummaryMemory(
            llm=get_llm_provider(provider_name=provider_name),
            chat_memory=get_session_message_history(session_id),
            return_messages=True,
        )

    return SESSION_MESSAGE_SUMMARY[session_id]

def get_session_active_memory(session_id: str):
    return SESSION_ACTIVE_MEMORY.get(session_id, [])

def update_session_active_memory(session_id: str, user_input: str, ai_output: str):
    if session_id not in SESSION_ACTIVE_MEMORY:
        SESSION_ACTIVE_MEMORY[session_id] = []

    SESSION_ACTIVE_MEMORY[session_id].append({"user": user_input, "ai": ai_output})

def get_active_memory_formatted(session_id: str) -> str:
    history = get_session_active_memory(session_id=session_id)

    if not history:
        return "No recent chat history available!"

    else:
        conversation = []

        for chat in history:
            conversation.append(f"User: {chat['user']}\nAssisstant: {chat['ai']}")

        return "\n\n".join(conversation)


def get_memory_summary(session_id: str, provider: str):
    return get_session_message_summary(session_id=session_id, provider_name=provider).load_memory_variables({}).get("summary", "")


def get_active_context(session_id: str) -> str:
    return get_active_memory_formatted(session_id)


def add_memory_to_rag_chain(session_id: str, provider: str, rag_chain, enabled: bool):
    if not enabled:
        print(f"🧠 Memory is not enabled. Stateless chat.")
        return rag_chain

    print(f"🧠 Hybrid memory is enabled for the session {session_id}")
    session_summary = get_session_message_summary(session_id=session_id, provider_name=provider)

    def rag_chain_with_summary(input_data, config):
        response = rag_chain.invoke(input_data, config)
        user_input = input_data["question"]
        ai_output = getattr(response, "content", str(response))

        # save session summary
        session_summary.save_context({"input": user_input}, {"output": ai_output})

        # update active summary
        update_session_active_memory(session_id=session_id, user_input=user_input, ai_output=ai_output)

        # refresh summary
        messages = session_summary.chat_memory.messages
        existing_summary = session_summary.load_memory_variables({}).get("summary", "")
        new_summary = session_summary.predict_new_summary(messages, existing_summary)
        session_summary._buffer = new_summary

        return response

    runnable_with_summary = RunnableLambda(rag_chain_with_summary)

    return RunnableWithMessageHistory(
        runnable_with_summary,
        get_session_history=lambda _: get_session_message_history(session_id=session_id),
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key=None,
    )
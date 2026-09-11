import os
import sys
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from source.ingestion import load_documents
from source.chunking import split_documents
from source.embedding import get_embeddings_provider

from source.vectordb import sync_documents_to_vector_db
from source.retriever import get_retriever

from source.rag_chain import build_rag_chain
from source.native_memory import add_memory_to_rag_chain

def stream_data(text: str):
    for word in text.split(" "):
        yield(word + " ")
        time.sleep(0.05)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = list()


if "active_session" not in st.session_state:
    st.session_state.active_session = dict()


st.set_page_config(
    page_title="Basic RAG Capstone Project",
    page_icon="🦜",
    layout="wide"
)

st.markdown("""
<style>
.body {
    background: #FFF;
    padding: 0;
    margin: 0;
}

.title-subheading {
    padding: 0 !important;
    margin: 0 !important;
    font-size: 16px !important;
    font-weight: 300 !important;
    margin-left: 54px !important;
    margin-bottom: 24px !important;
}

.sidebar-header-placeholder {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
    margin-bottom: 32px;
}

.sidebar-header-placeholder .icon {
    font-family: 'Material Symbols Rounded';
    vertical-align: middle;
    color: steelblue;
    font-size: 42px;
}

.sidebar-header-placeholder .content {
    display: flex;
    flex-direction: column;
}

.sidebar-header-placeholder .content h1 {
    margin: 0;
    padding: 0;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 4px;
}

.sidebar-header-placeholder .content h2 {
    margin: 0;
    padding: 0;
    font-size: 12px;
    font-weight: normal;
    margin-top: 4px;
}

.st-key-select-chat-session .stSelectbox p {
    padding: 18px 0 8px;
    font-weight: bold;
    font-size: 16px;
}

.st-key-select-chat-session .stSelectbox p span {
    font-size: 24px !important;
    color: cornflowerblue !important;
    vertical-align: middle !important;
}

span.stMarkdownBadge {
    padding: 4px 8px !important;
    margin-bottom: 16px !important;
    border-radius: 0.5rem !important;
}

span.model-engine-title {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 0px;
}

span.model-engine-title span {
    color: deeppink !important;
    font-size: 24px !important;
    vertical-align: middle !important;
}

[class*="st-key-llm-provider"] {
    margin-bottom: 16px !important;
}

.upload-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #4B5563;
    margin-bottom: 0.5rem;
}

.llm-thinking {
    display: flex;
    flex-direction: row;
    justify-content: flex-start;
    align-items: center;
    margin-top: -16px;
    gap: 16px;
    font-weight: 200;
    font-style: italic;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.header("⚡Cognitive RAG Engine")
st.markdown("""<h2 class="title-subheading">Intelligent Knowledge Retrieval & Synthesis Powered by Advanced Vector Search</h2>""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-header-placeholder">
    <div class="icon">settings</div>
    <div class="content">
        <h1>Session & Model Settings</h1>
        <h2>Managing sessions and model engine</h2>
    </div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("Start New Session", icon=":material/chat_add_on:", width="stretch", type="primary"):
    new_id = str(uuid.uuid4())
    new_name = f"Chat Session - {len(st.session_state.chat_sessions) + 1}"
    chat_session = {
        "_id": new_id,
        "name": new_name,
        "llm_provider": "OPENAI",
        "memory_enabled": False,
        "chat_history": []
    }
    st.session_state.chat_sessions.append(chat_session)
    st.session_state["select-chat-session"] = new_name
    st.session_state.active_session = chat_session

if not len(st.session_state.chat_sessions):
    new_id = str(uuid.uuid4())
    new_name = "Chat Session - 1"
    chat_session = {
        "_id": new_id,
        "name": new_name,
        "llm_provider": "OPENAI",
        "memory_enabled": False,
        "chat_history": []
    }
    st.session_state.chat_sessions.append(chat_session)
    st.session_state.active_session = chat_session

chat_session_names: list[str] = [session["name"] for session in st.session_state.chat_sessions]

selected_chat_session = st.sidebar.selectbox(
    ":material/chat: Active Chat Session",
    key="select-chat-session",
    options=chat_session_names,
    index=chat_session_names.index(st.session_state.active_session.get("name", ''))
)

for session in st.session_state.chat_sessions:
    if session["name"] == selected_chat_session:
        st.session_state.active_session = session
        break


st.sidebar.badge(f"**ID: {st.session_state.active_session.get("_id").split("-")[-1:][0]}**", color="violet")


st.sidebar.markdown("""
<span class="model-engine-title">:material/neurology: Model Engine</span>
""", unsafe_allow_html=True)


st.session_state.active_session["llm_provider"] = st.sidebar.selectbox(
    "LLM Provider",
    options=["OPENAI", "GEMINI", "GROQ"],
    key=f"llm-provider-{(st.session_state.active_session.get("name", "default")).replace(" ", "-")}",
    index=["OPENAI", "GEMINI", "GROQ"].index(st.session_state.active_session.get("llm_provider", "OPENAI"))
)


st.session_state.active_session["memory_enabled"] = st.sidebar.toggle(
    "Enable Context Memory",
    value=st.session_state.active_session["memory_enabled"],
    key=f"memory-toggle-{(st.session_state.active_session.get("name", "Chat Session - 1")).replace(" ", "-")}"
)

st.sidebar.divider(width="stretch")


with st.expander(":material/note_stack: **Knowledge Base & File Hub**", expanded=st.session_state.active_session.get("expander", True)):

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Indexed Documents", value=st.session_state.active_session.get("indexed_document_chunks", 0))
        
    with col2:
        st.metric(label="Active Chat Session", value=f"{st.session_state.active_session.get("name", "Chat Session - 1")}")

    st.divider()

    st.markdown("<div class='upload-label'>Upload Documents to Enrich Vector Memory</div>", unsafe_allow_html=True)

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    uploaded_document = st.file_uploader(
        disabled=st.session_state.active_session.get("disabled_uploader", False),
        key=f"uploader_{st.session_state.file_uploader_key}",
        label="Upload Documents to Enrich Vector Memory",
        type=["pdf", "docx", "docs", "doc", "txt", "md"],
        label_visibility="collapsed",
        accept_multiple_files=False,
        width="stretch"
    )

    if uploaded_document:
        with st.spinner("Ingesting and parsing documents for VectorDB embedding and indexing.", show_time=True):
            _id = st.session_state.active_session.get("_id", uuid.uuid4())
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", f"{_id}_{uploaded_document.name}"))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file=file_path, mode="wb") as f:
                f.write(uploaded_document.getbuffer())

            try:
                name, ext = os.path.splitext(file_path)
                ext = ext.lower()
                if ext == ".pdf":
                    file_icon = ":material/picture_as_pdf:"
                elif ext in [".txt", ".md"]:
                    file_icon = ""
                elif ext in [".docx", ".docs", ".doc"]:
                    file_icon = ""
                
                docs = load_documents(path=file_path)
                chunks = split_documents(documents=docs, chunk_size=1000, chunk_overlap=200)
                embedding = get_embeddings_provider(provider_name=st.session_state.active_session["llm_provider"])

                vectorDB = sync_documents_to_vector_db(collection_name=st.session_state.active_session["_id"], documents=chunks, embedding_engine=embedding)

                st.session_state.active_session["indexed_document_filename"] = uploaded_document.name
                st.session_state.active_session["indexed_document_chunks"] = len(chunks)
                st.session_state.active_session["disabled_uploader"] = True
                st.session_state.active_session["expander"] = False

                if os.path.exists(path=file_path):
                    os.remove(path=file_path)

                st.session_state.file_uploader_key += 1
                st.rerun()

            except Exception as e:
                st.error(f"We got an exception -> {e}", icon=":material/error:")
                st.stop()


    if "indexed_document_filename" in st.session_state.active_session:
        st.success(f"VectorDB processed and indexed for - {st.session_state.active_session.get("indexed_document_filename", "")}")

        st.markdown(
            f"""
            **Active file in context**
            <br />
            <span style="padding: 0 4px 0 0;">:material/description:</span> {st.session_state.active_session.get("indexed_document_filename", "")}
            """,
            unsafe_allow_html=True
        )

if "indexed_document_filename" in st.session_state.active_session and str(st.session_state.active_session["llm_provider"]).lower() == "groq":
    st.warning(":yellow[:material/warning:] Expect slower response when selecting **groq**")

if "indexed_document_filename" in st.session_state.active_session:
    try:
        embedding = get_embeddings_provider(provider_name=st.session_state.active_session["llm_provider"])

        vectorDB = sync_documents_to_vector_db(collection_name=st.session_state.active_session["_id"], documents=[], embedding_engine=embedding)

        retriever = get_retriever(vectordb=vectorDB, provider_name=st.session_state.active_session["llm_provider"])

        rag_chain = build_rag_chain(retriever=retriever, provider_name=st.session_state.active_session["llm_provider"])

        rag_chain_with_memory = add_memory_to_rag_chain(
            rag_chain=rag_chain,
            enabled=st.session_state.active_session["memory_enabled"],
            provider=st.session_state.active_session["llm_provider"],
            session_id=st.session_state.active_session["_id"],
        )

    except Exception as e:
        st.error(f"Error initializing backend: {e}")
        st.stop()
    
    icons = {
        "openai": "https://img.icons8.com/fluency-systems-regular/64/chatgpt.png",
        "gemini": "https://img.icons8.com/skeuomorphism/64/gemini-ai.png",
        "groq": "https://unpkg.com/@lobehub/icons-static-svg@latest/icons/groq.svg"
    }

    provider_key = str(st.session_state.active_session["llm_provider"]).lower()
    assistant_avatar = icons[provider_key]
    user_avatar = "https://img.icons8.com/retro/64/user.png"


    for chat in st.session_state.active_session["chat_history"]:
        with st.chat_message(
            chat["role"],
            avatar=assistant_avatar if chat["role"] == "assistant" else user_avatar
        ):
            st.markdown(chat["content"])


    if prompt := st.chat_input("Ask questions about your ingested document"):
        st.session_state.active_session["chat_history"].append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=assistant_avatar):
            assistant_placeholder = st.empty()

            with assistant_placeholder.container():
                st.markdown(
                    """
                    <div class="llm-thinking">
                        <span>Retrieving from the most relevant result</span>
                        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48Y2lyY2xlIGN4PSI0IiBjeT0iMTIiIHI9IjMiIGZpbGw9IiM4YjhiOGIiPjxhbmltYXRlIGlkPSJTVkc3eDE0RGNvbSIgZmlsbD0iZnJlZXplIiBhdHRyaWJ1dGVOYW1lPSJvcGFjaXR5IiBiZWdpbj0iMDtTVkdxU2pHMGRVcC5lbmQtMC4yNXMiIGR1cj0iMC43NXMiIHZhbHVlcz0iMTsuMiIvPjwvY2lyY2xlPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjMiIGZpbGw9IiM4YjhiOGIiIG9wYWNpdHk9Ii40Ij48YW5pbWF0ZSBmaWxsPSJmcmVlemUiIGF0dHJpYnV0ZU5hbWU9Im9wYWNpdHkiIGJlZ2luPSJTVkc3eDE0RGNvbS5iZWdpbiswLjE1cyIgZHVyPSIwLjc1cyIgdmFsdWVzPSIxOy4yIi8+PC9jaXJjbGU+PGNpcmNsZSBjeD0iMjAiIGN5PSIxMiIgcj0iMyIgZmlsbD0iIzhiOGI4YiIgb3BhY2l0eT0iLjMiPjxhbmltYXRlIGlkPSJTVkdxU2pHMGRVcCIgZmlsbD0iZnJlZXplIiBhdHRyaWJ1dGVOYW1lPSJvcGFjaXR5IiBiZWdpbj0iU1ZHN3gxNERjb20uYmVnaW4rMC4zcyIgZHVyPSIwLjc1cyIgdmFsdWVzPSIxOy4yIi8+PC9jaXJjbGU+PC9zdmc+" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            full_response = ""

            st.session_state.active_session["chat_history"].append({"role": "assistant", "content": full_response})
            assistant_index = len(st.session_state.active_session["chat_history"]) - 1

            try:
                response = rag_chain_with_memory.invoke(
                    {"question": prompt},
                    config={"configurable": {"session_id": st.session_state.active_session["_id"]}},
                )
                full_response = getattr(response, "content", str(response))
                if (provider_key == "gemini"):
                    full_response = full_response[0]["text"]

                st.session_state.active_session["chat_history"][assistant_index]["content"] = full_response

                assistant_placeholder.write_stream(stream_data(full_response))

            except Exception as e:
                full_response = f":red[:material/error:] Error: {e}"
                st.session_state.active_session["chat_history"][assistant_index]["content"] = full_response
                assistant_placeholder.error(stream_data(full_response))

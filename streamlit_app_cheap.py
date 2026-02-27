import os
from pathlib import Path

import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage

# ---------- Streamlit config ----------
st.set_page_config(page_title="Buddhism Info Bot", page_icon="🧘", layout="centered")


# ---------- Helpers ----------
def get_secret(name: str, default: str | None = None) -> str | None:
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vector_index"
EMBED_CACHE_DIR = BASE_DIR / ".cache" / "embeddings"

TOP_K = 3  # <-- fixed value, no slider

# ---------- Sidebar ----------
st.sidebar.header("Controls")
if st.sidebar.button("Reset conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ---------- UI ----------
st.title("Buddhism Info Bot")
st.write(
    """
Ask a question about Buddhism.  
The bot answers **only from the indexed documents** and speaks like a **young Tibetan professor**.
"""
)

# ---------- Per-user session memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ---------- Cached shared resources ----------
@st.cache_resource
def make_llm() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it in Streamlit Secrets (TOML).")
    return Groq(model="llama-3.3-70b-versatile", api_key=api_key)


@st.cache_resource
def make_embeddings() -> HuggingFaceEmbedding:
    embedding_model = get_secret(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbedding(
        model_name=embedding_model, cache_folder=str(EMBED_CACHE_DIR)
    )


@st.cache_resource
def load_vector_index():
    if not VECTOR_DIR.exists():
        raise RuntimeError(
            f"Missing '{VECTOR_DIR.name}' folder in the repo. "
            "Commit your persisted LlamaIndex storage (vector_index/) to GitHub."
        )
    embeddings = make_embeddings()
    storage_context = StorageContext.from_defaults(persist_dir=str(VECTOR_DIR))
    return load_index_from_storage(storage_context, embed_model=embeddings)


def build_chat_engine():
    """
    Build a fresh chat engine for this request.
    We reset memory each time and pass only this user's history (template behavior).
    """
    llm = make_llm()
    index = load_vector_index()

    retriever = index.as_retriever(similarity_top_k=TOP_K)
    memory = ChatMemoryBuffer.from_defaults(token_limit=2500)

    context_prompt = (
        "You are a young Tibetan professor. Speak calmly, clearly, and thoughtfully.\n"
        "You must answer using ONLY the provided context.\n"
        "If the context does not contain the answer, say you cannot find it in the provided texts.\n"
        "Here is the relevant context:\n"
        "{context_str}\n"
    )

    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        memory=memory,
        retriever=retriever,
        context_prompt=context_prompt,
        verbose=False,
    )


def stream_and_collect(response):
    collected = {"text": ""}

    def gen():
        for token in response.response_gen:
            collected["text"] += token
            yield token

    return gen(), collected


# ---------- Chat input (auto-clears) ----------
prompt = st.chat_input("Ask your question…")

if prompt:
    # Store and show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert this user's session history to LlamaIndex ChatMessage list
    bot_history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]

    try:
        chat_engine = build_chat_engine()
        chat_engine.reset()

        with st.chat_message("assistant"):
            with st.spinner("Consulting the scriptures…"):
                response = chat_engine.stream_chat(prompt, chat_history=bot_history)

            token_gen, collected = stream_and_collect(response)
            st.write_stream(token_gen)

        # Persist assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": collected["text"]}
        )

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(str(e))

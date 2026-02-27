import os
from pathlib import Path

import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate


# ---------- Streamlit config ----------
st.set_page_config(page_title="Buddhism Info Bot", page_icon="🧘", layout="centered")


# ---------- Helpers ----------
def get_secret(name: str, default: str | None = None) -> str | None:
    # Streamlit Community Cloud: define secrets in the app settings
    # Local dev: you can also export env vars (or use .streamlit/secrets.toml)
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vector_index"  # <-- commit this folder to GitHub
EMBED_CACHE_DIR = (
    BASE_DIR / ".cache" / "embeddings"
)  # runtime cache on cloud (ephemeral)


# ---------- UI ----------
st.title("Buddhism Info Bot")

st.write(
    """
Ask a question about Buddhism.  
The bot answers **only from the indexed documents** and speaks like a **wise Tibetan monk**.
"""
)

question = st.text_input("What is your question?")


# ---------- Resources (cached) ----------
@st.cache_resource
def make_llm() -> Groq:
    api_key = st.secrets["GROQ_API_KEY"]
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it in Streamlit → App settings → Secrets."
        )
    return Groq(model="llama-3.3-70b-versatile", api_key=api_key)


@st.cache_resource
def make_embeddings() -> HuggingFaceEmbedding:
    # NOTE: Qwen3-Embedding-0.6B is quite heavy for Streamlit Cloud RAM.
    # If the app crashes, switch to a smaller model and rebuild your index with it.
    # switched to sentence-transformers/all-MiniLM-L6-v2
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


@st.cache_resource
def make_query_engine():
    llm = make_llm()
    index = load_vector_index()

    monk_qa_tmpl = PromptTemplate(
        """Context information is below.
---------------------
{context_str}
---------------------

Answer the question using only the context.
Speak like a Tibetan person with a university degree (speak like a young professor from Tibet).

Query: {query_str}
Answer:"""
    )

    monk_refine_tmpl = PromptTemplate(
        """The original query is: {query_str}

We already have an answer:
{existing_answer}

We have more context below:
------------
{context_msg}
------------

Refine the answer if needed, using only the new context.
Keep the professor speaking style.
If the new context is not useful, return the original answer unchanged.

Refined Answer:"""
    )

    # streaming=True gives you a token generator you can pass to st.write_stream(...)
    return index.as_query_engine(
        llm=llm,
        text_qa_template=monk_qa_tmpl,
        similarity_top_k=2,
        response_mode="compact",
        refine_template=monk_refine_tmpl,
        streaming=True,
    )


# ---------- Run ----------
if question:
    try:
        query_engine = make_query_engine()
        response = query_engine.query(question)

        # Stream if available, otherwise fall back to plain text
        if hasattr(response, "response_gen") and response.response_gen is not None:
            st.write_stream(response.response_gen)
        else:
            st.write(str(response))

    except Exception as e:
        st.error(str(e))
else:
    st.info("Type a question to begin.")

"""
Brag & Bev AI Agent — Query-only RAG on Render.
Uses prebuilt Chroma at data/chroma. No uploads, no ingestion. Smoke mode: ?smoke=1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Page config first
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Brag & Bev AI Agent",
    page_icon="📄",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Smoke-test mode
# -----------------------------------------------------------------------------
def _smoke_requested() -> bool:
    try:
        q = getattr(st, "query_params", None)
        if q is None:
            return False
        val = (q.get("smoke") or "") if callable(getattr(q, "get", None)) else ""
        if isinstance(val, (list, tuple)):
            val = (val[0] or "") if val else ""
        return str(val).strip().lower() in ("1", "true", "yes")
    except Exception:
        return False

if _smoke_requested():
    st.success("Smoke test OK — Streamlit is running.")
    st.caption("Add ?smoke=1 to the URL. No embeddings, Chroma, or Groq loaded.")
    st.code(f"Python {sys.version}\nStreamlit {st.__version__}", language="text")
    st.stop()

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
RETRIEVAL_K = 5

# -----------------------------------------------------------------------------
# Prebuilt DB check — lightweight, no embeddings yet
# -----------------------------------------------------------------------------
def _chroma_available() -> bool:
    p = Path(CHROMA_DIR)
    if not p.exists() or not p.is_dir():
        return False
    # Chroma persists chroma.sqlite3 and collection dirs
    if (p / "chroma.sqlite3").exists():
        return True
    # Alternative layout: list dir and ensure non-empty
    return any(p.iterdir())

if not _chroma_available():
    st.error("No prebuilt vector database found in data/chroma. Build it locally first.")
    st.caption("Run indexing locally (e.g. index_docs.py), commit data/chroma, then redeploy.")
    st.stop()

# -----------------------------------------------------------------------------
# Lazy imports (only after smoke + chroma check)
# -----------------------------------------------------------------------------
try:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.messages import HumanMessage
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Cached resources — created only on first Run AI
# -----------------------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Set it in Render Dashboard → Environment (or .env locally)."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        api_key=api_key,
    )

# -----------------------------------------------------------------------------
# Prompt + RAG
# -----------------------------------------------------------------------------
def build_prompt(question: str, retrieved_chunks: list) -> str:
    context_block = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in retrieved_chunks
    )
    return f"""You are the Brag & Bev AI Agent.

Answer the user's question using the provided context.
If the answer is not in the documents, say it is not available in the knowledge base.

User Question:
{question}

Context:
{context_block}

Answer:"""

def run_rag(question: str, k: int = RETRIEVAL_K):
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        return f"Vectorstore error: {e}", []
    try:
        docs = vectorstore.similarity_search(question, k=k)
    except Exception as e:
        return f"Retrieval error: {e}", []
    if not docs:
        return (
            "No documents in the knowledge base or none match your question.",
            [],
        )
    try:
        llm = get_llm()
    except ValueError as e:
        return str(e), docs
    prompt = build_prompt(question, docs)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"LLM error: {e}", docs
    return answer.strip(), docs

# -----------------------------------------------------------------------------
# UI — render immediately
# -----------------------------------------------------------------------------
st.title("Brag & Bev AI Agent")
st.caption("Query the knowledge base. Uses prebuilt data/chroma (no uploads).")

question = st.text_input("Ask a question", placeholder="e.g. What is the dumpster diver?")

if st.button("Run AI", type="primary"):
    if not (question and question.strip()):
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, retrieved = run_rag(question.strip())
                st.subheader("Answer")
                st.write(answer)
                with st.expander("Retrieved context"):
                    if retrieved:
                        for i, doc in enumerate(retrieved, 1):
                            src = doc.metadata.get("source", "unknown")
                            st.markdown(f"**{i}. {src}**")
                            st.text(doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""))
                            st.divider()
                    else:
                        st.write("No chunks retrieved.")
            except Exception as e:
                st.error(str(e))

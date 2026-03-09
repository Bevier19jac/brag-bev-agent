"""
Brag & Bev AI Agent — Lightweight query-only RAG on Render.
Reads data/raw (.txt, .md, .json, .csv), TF-IDF retrieval, Groq for answers.
No Chroma, no embeddings, no file upload. Smoke mode: ?smoke=1
"""
from __future__ import annotations

import os
import sys
import json
import csv
from io import StringIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Page config first (must be first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Brag & Bev AI Agent",
    page_icon="📄",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Smoke-test mode: render immediately, no heavy imports
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
    st.caption("Add ?smoke=1 to the URL. No TF-IDF or Groq loaded.")
    st.code(f"Python {sys.version}\nStreamlit {st.__version__}", language="text")
    st.stop()

# -----------------------------------------------------------------------------
# Constants — no heavy init at import
# -----------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
GROQ_MODEL = "llama-3.3-70b-versatile"

# -----------------------------------------------------------------------------
# Load text from data/raw (only .txt, .md, .json, .csv)
# -----------------------------------------------------------------------------
def _read_file(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return None
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return None
    if suffix == ".json":
        try:
            obj = json.loads(text)
            return json.dumps(obj, indent=2)
        except Exception:
            return text
    if suffix == ".csv":
        try:
            reader = csv.reader(StringIO(text))
            return "\n".join(" | ".join(row) for row in reader)
        except Exception:
            return text
    return text

def load_documents_from_raw():
    """Return list of (source_name, text). Empty if data/raw missing or no supported files."""
    if not RAW_DIR.exists() or not RAW_DIR.is_dir():
        return []
    out = []
    for path in RAW_DIR.rglob("*"):
        if not path.is_file():
            continue
        text = _read_file(path)
        if text and text.strip():
            out.append((path.name, text.strip()))
    return out

# -----------------------------------------------------------------------------
# Chunk text (simple fixed-size with overlap)
# -----------------------------------------------------------------------------
def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Yield (chunk_text, source) for each chunk."""
    if not text or not text.strip():
        return
    start = 0
    text = text.replace("\r\n", "\n")
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_br = chunk.rfind("\n")
            if last_br > chunk_size // 2:
                chunk = chunk[: last_br + 1]
                end = start + len(chunk)
        if chunk.strip():
            yield (chunk.strip(), source)
        start = end - overlap if overlap < end - start else end

def build_chunks(docs: list[tuple[str, str]]):
    """Return list of dicts: {"text": str, "source": str}."""
    chunks = []
    for source_name, text in docs:
        for chunk_text_val, source in chunk_text(text, source_name):
            chunks.append({"text": chunk_text_val, "source": source})
    return chunks

# -----------------------------------------------------------------------------
# TF-IDF retrieval — lazy import sklearn only when Run AI is used
# -----------------------------------------------------------------------------
def run_tfidf_retrieval(chunks: list[dict], question: str, k: int = TOP_K):
    """Return top-k chunks by TF-IDF cosine similarity. Uses sklearn."""
    if not chunks or not question.strip():
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        raise ImportError(f"scikit-learn required for retrieval: {e}") from e
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    try:
        X = vectorizer.fit_transform(texts)
    except Exception as e:
        raise RuntimeError(f"TF-IDF fit failed: {e}") from e
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, X).ravel()
    top_indices = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]

# -----------------------------------------------------------------------------
# Groq LLM — lazy import
# -----------------------------------------------------------------------------
def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Set it in Render Dashboard → Environment (or .env locally)."
        )
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        api_key=api_key,
    ), HumanMessage

def build_prompt(question: str, retrieved: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved
    )
    return f"""You are the Brag & Bev AI Agent.

Answer the user's question using the provided context from company documents.
If the answer is not in the context, say it is not available in the knowledge base.

User Question:
{question}

Context:
{context_block}

Answer:"""

def run_rag(question: str, k: int = TOP_K):
    """Load data/raw -> chunk -> TF-IDF retrieve -> Groq -> (answer, retrieved_chunks)."""
    docs = load_documents_from_raw()
    if not docs:
        return "No source documents found in data/raw.", []
    chunks = build_chunks(docs)
    if not chunks:
        return "No source documents found in data/raw.", []
    try:
        retrieved = run_tfidf_retrieval(chunks, question, k=k)
    except Exception as e:
        return f"Retrieval error: {e}", []
    if not retrieved:
        return "No matching chunks for your question.", []
    try:
        llm, HumanMessage = get_llm()
    except ValueError as e:
        return str(e), retrieved
    prompt = build_prompt(question, retrieved)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"LLM error: {e}", retrieved
    return answer.strip(), retrieved

# -----------------------------------------------------------------------------
# UI — render immediately; data/raw check only when Run AI is clicked
# -----------------------------------------------------------------------------
st.title("Brag & Bev AI Agent")
st.caption("Query-only: reads documents from data/raw, TF-IDF retrieval, Groq for answers.")

# Optional: show data/raw status in sidebar
with st.sidebar:
    st.header("Source")
    if RAW_DIR.exists() and RAW_DIR.is_dir():
        docs = load_documents_from_raw()
        if docs:
            st.success(f"Found {len(docs)} file(s) in data/raw.")
            for name, _ in docs:
                st.caption(f"• {name}")
        else:
            st.warning("No supported files (.txt, .md, .json, .csv) in data/raw.")
    else:
        st.error("data/raw directory not found.")

question = st.text_input("Ask a question", placeholder="e.g. What is the dumpster diver?")

if st.button("Run AI", type="primary"):
    if not (question and question.strip()):
        st.warning("Please enter a question.")
    else:
        with st.spinner("Loading documents, retrieving with TF-IDF, generating answer..."):
            try:
                answer, retrieved = run_rag(question.strip())
                st.subheader("Answer")
                st.write(answer)
                with st.expander("Retrieved context"):
                    if retrieved:
                        for i, c in enumerate(retrieved, 1):
                            st.markdown(f"**{i}. {c['source']}**")
                            st.text(c["text"][:800] + ("..." if len(c["text"]) > 800 else ""))
                            st.divider()
                    else:
                        st.write("No chunks retrieved.")
            except Exception as e:
                st.error(str(e))

"""
Brag & Bev AI Agent — Production RAG app for Render.
Streamlit + Chroma + Groq. No deprecated imports. Lazy init; smoke-test mode.
"""
from __future__ import annotations

import os
import sys
import json
import csv
from datetime import datetime, timezone
from io import StringIO, BytesIO
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
# Smoke-test mode: render immediately without touching embeddings/Chroma/Groq
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
    st.caption("Add ?smoke=1 to the URL to see this page. No embeddings, Chroma, or Groq were loaded.")
    st.code(f"Python {sys.version}\nStreamlit {st.__version__}", language="text")
    st.stop()

# -----------------------------------------------------------------------------
# Lazy imports (only after smoke check) — any failure will be caught below
# -----------------------------------------------------------------------------
try:
    from pypdf import PdfReader
    from docx import Document as DocxDocument
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.messages import HumanMessage
except ImportError as e:
    st.error(f"Import error — fix dependencies: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
APP_TITLE = "Brag & Bev AI Agent"
DATA_DIR = "data"
RAW_DIR = "data/raw"
CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".json", ".md"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5

# -----------------------------------------------------------------------------
# Ensure dirs (non-throwing)
# -----------------------------------------------------------------------------
def _ensure_dirs():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

_ensure_dirs()

# -----------------------------------------------------------------------------
# Cached resources — only created when first needed; failures surface in UI
# -----------------------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return emb

@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    vs = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vs

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Set it in Render Dashboard → Environment (or in .env locally)."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        api_key=api_key,
    )

# -----------------------------------------------------------------------------
# File handling
# -----------------------------------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    path = Path(RAW_DIR) / uploaded_file.name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(path)

def extract_text_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")

def extract_text_md(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")

def extract_text_json(content: bytes) -> str:
    try:
        obj = json.loads(content.decode("utf-8", errors="replace"))
        return json.dumps(obj, indent=2)
    except Exception:
        return content.decode("utf-8", errors="replace")

def extract_text_csv(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    reader = csv.reader(StringIO(decoded))
    return "\n".join(" | ".join(row) for row in reader)

def extract_text_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        parts.append(text or "")
    return "\n".join(parts)

def extract_text_docx(content: bytes) -> str:
    doc = DocxDocument(BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())

def extract_text_from_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    raw = uploaded_file.getvalue()
    if suffix == ".txt":
        return extract_text_txt(raw)
    if suffix == ".md":
        return extract_text_md(raw)
    if suffix == ".json":
        return extract_text_json(raw)
    if suffix == ".csv":
        return extract_text_csv(raw)
    if suffix == ".pdf":
        return extract_text_pdf(raw)
    if suffix == ".docx":
        return extract_text_docx(raw)
    raise ValueError(f"Unsupported file type: {suffix}")

# -----------------------------------------------------------------------------
# Chunking and ingestion
# -----------------------------------------------------------------------------
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def ingest_files(uploaded_files):
    if not uploaded_files:
        return [], 0
    try:
        embeddings = get_embeddings()
        vectorstore = get_vectorstore()
    except Exception as e:
        st.error(f"Cannot load embeddings or vectorstore: {e}")
        return [], 0
    splitter = get_text_splitter()
    processed = []
    total_chunks = 0
    for uf in uploaded_files:
        try:
            text = extract_text_from_file(uf)
        except ValueError as e:
            st.warning(f"Skipping {uf.name}: {e}")
            continue
        if not (text and text.strip()):
            st.warning(f"No readable text in {uf.name}. Skipping.")
            continue
        try:
            save_uploaded_file(uf)
        except Exception as e:
            st.warning(f"Could not save {uf.name} to {RAW_DIR}: {e}")
            continue
        try:
            chunks = splitter.split_text(text)
            metadatas = [{"source": uf.name, "chunk_idx": i} for i in range(len(chunks))]
            vectorstore.add_texts(texts=chunks, metadatas=metadatas)
            total_chunks += len(chunks)
            processed.append(uf.name)
            st.sidebar.caption(f"Ingested: {uf.name} ({len(chunks)} chunks)")
        except Exception as e:
            st.error(f"Ingestion failed for {uf.name}: {e}")
    return processed, total_chunks

# -----------------------------------------------------------------------------
# Prompt and RAG (similarity_search + custom prompt, no RetrievalQA)
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
            "No documents in the knowledge base or none match your question. "
            "Upload and ingest files first, then try again.",
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
# UI
# -----------------------------------------------------------------------------
# Rerun counter (increment every script run to detect unexpected reruns)
if "_rerun_count" not in st.session_state:
    st.session_state._rerun_count = 0
st.session_state._rerun_count += 1

st.title(APP_TITLE)
st.caption("Upload documents in the sidebar, ingest into the knowledge base, then ask questions. Answers are grounded in your files.")

# Debug toggle (visible)
with st.sidebar:
    # Timestamp and rerun counter to see if the app is rerunning unexpectedly
    st.caption(f"Rerun #{st.session_state._rerun_count} — {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

    uploaded = None
    show_debug = False
    try:
        st.header("Knowledge Base")
        show_debug = st.checkbox("Show debug messages", value=False, help="Show when embeddings/DB load and retrieval counts")
        if show_debug and st.session_state.get("vectorstore_used"):
            st.caption("Vector DB used successfully (embeddings + Chroma loaded).")
        smoke_url = st.text_input("Smoke test URL", value="", help="Copy URL and add ?smoke=1 to test Streamlit only")
        if smoke_url:
            st.caption("Open: " + (smoke_url + ("&" if "?" in smoke_url else "?") + "smoke=1"))

        uploaded = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "docx", "csv", "json", "md"],
            accept_multiple_files=True,
            help="PDF, TXT, DOCX, CSV, JSON, Markdown",
        )

        # Uploader diagnostic (immediately after file_uploader)
        if uploaded is None:
            st.warning("Uploader returned None")
        elif isinstance(uploaded, (list, tuple)) and len(uploaded) == 0:
            st.warning("Uploader returned empty list")
        elif uploaded:
            st.success(f"Uploader: {len(uploaded)} file(s)")
            for i, f in enumerate(uploaded):
                st.caption(f"  • {getattr(f, 'name', repr(f))}")

        if st.button("Ingest uploaded files", type="primary", use_container_width=True):
            if not uploaded:
                st.warning("Upload at least one file first.")
            else:
                with st.spinner("Extracting text, chunking, storing in Chroma..."):
                    try:
                        processed, num_chunks = ingest_files(uploaded)
                        if processed:
                            st.success(f"Ingested {len(processed)} file(s), {num_chunks} chunks.")
                            if show_debug:
                                st.info(f"Documents ingested; total chunks from this batch: {num_chunks}")
                            st.session_state.vectorstore_used = True
                        else:
                            st.info("No files could be processed. Check format and try again.")
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

        st.divider()
        st.caption("Raw files: data/raw. Vectors: data/chroma.")
    except Exception as e:
        st.error(f"Sidebar upload block error: {e}")

# Main: question + RAG
question = st.text_input("Ask a question about your documents", placeholder="e.g. What is the dumpster diver?")

if st.button("Run AI", type="primary"):
    if not (question and question.strip()):
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, retrieved = run_rag(question.strip())
                if show_debug and retrieved is not None:
                    st.info(f"Retrieval returned {len(retrieved)} doc(s).")
                if retrieved:
                    st.session_state.vectorstore_used = True
                if not retrieved and "No documents" in answer and show_debug:
                    st.warning("Retrieval returned 0 docs.")
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
                st.error(f"Error: {e}")

"""
Brag & Bev AI Agent — Production-ready RAG document agent.
Streamlit UI: upload PDF/TXT/DOCX/CSV/JSON/MD, ingest into Chroma, query with Groq.
No deprecated LangChain chains; uses similarity_search + custom prompt + ChatGroq.
"""
import os
import json
import csv
from io import StringIO, BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

load_dotenv()

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
# Ensure directories exist
# -----------------------------------------------------------------------------
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
st.title(APP_TITLE)
st.caption("Upload documents, ingest into the knowledge base, then ask questions. Answers are grounded in your uploaded files.")


# -----------------------------------------------------------------------------
# Cached resources (embeddings + vectorstore)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


# -----------------------------------------------------------------------------
# LLM (not cached — checks env each time for clear error when missing)
# -----------------------------------------------------------------------------
def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it in Render Dashboard → Environment (or in .env locally)."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        api_key=api_key,
    )


# -----------------------------------------------------------------------------
# File handling: save to data/raw and extract text
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
        raise ValueError(f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")
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
    embeddings = get_embeddings()
    vectorstore = get_vectorstore()
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
        save_uploaded_file(uf)
        chunks = splitter.split_text(text)
        metadatas = [{"source": uf.name, "chunk_idx": i} for i in range(len(chunks))]
        vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        total_chunks += len(chunks)
        processed.append(uf.name)
    return processed, total_chunks


# -----------------------------------------------------------------------------
# Prompt (exact format required)
# -----------------------------------------------------------------------------
def build_prompt(question: str, retrieved_chunks: list) -> str:
    context_block = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in retrieved_chunks
    )
    return f"""You are the Brag & Bev AI Agent.

Answer the user's question using the provided context.
If the answer is not present in the documents, say that the information is not available in the uploaded knowledge base.

User Question:
{question}

Context:
{context_block}

Answer:"""


# -----------------------------------------------------------------------------
# Retrieval + LLM (no RetrievalQA; similarity_search + custom prompt + direct call)
# -----------------------------------------------------------------------------
def run_rag(question: str, k: int = RETRIEVAL_K):
    vectorstore = get_vectorstore()
    try:
        docs = vectorstore.similarity_search(question, k=k)
    except Exception as e:
        return f"Retrieval error: {e}", []
    if not docs:
        return (
            "No documents are in the knowledge base yet, or none match your question. "
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
# Sidebar: upload + ingest
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Knowledge Base")
    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx", "csv", "json", "md"],
        accept_multiple_files=True,
        help="PDF, TXT, DOCX, CSV, JSON, Markdown",
    )
    if st.button("Ingest uploaded files", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Upload at least one file first.")
        else:
            with st.spinner("Extracting text, chunking, and storing in Chroma..."):
                try:
                    processed, num_chunks = ingest_files(uploaded)
                    if processed:
                        st.success(f"Ingested {len(processed)} file(s), {num_chunks} chunks.")
                        for name in processed:
                            st.caption(f"• {name}")
                    else:
                        st.info("No files could be processed. Check format and try again.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption("Raw files are saved under data/raw. Vectors persist in data/chroma.")


# -----------------------------------------------------------------------------
# Main: question + run AI
# -----------------------------------------------------------------------------
question = st.text_input("Ask a question about your documents", placeholder="e.g. What is the dumpster diver?")

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
                st.error(f"Error: {e}")

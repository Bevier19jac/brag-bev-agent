import os
import json
import csv
from io import StringIO, BytesIO
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "Brag & Bev AI Agent"
PERSIST_DIR = "data/chroma"
UPLOAD_DIR = "data/raw"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"  # good default on Groq


# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload files, build the knowledge base, and ask questions about your documents.")


# -----------------------------
# ENSURE DIRECTORIES EXIST
# -----------------------------
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


# -----------------------------
# HELPERS
# -----------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )


def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment.")
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL,
        temperature=0.2,
    )


def save_uploaded_file(uploaded_file):
    file_path = Path(UPLOAD_DIR) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_md(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_json(file_bytes: bytes) -> str:
    try:
        obj = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        return json.dumps(obj, indent=2)
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_csv(file_bytes: bytes) -> str:
    decoded = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(StringIO(decoded))
    rows = [" | ".join(row) for row in reader]
    return "\n".join(rows)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    reader = PdfReader(BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    if suffix in [".txt"]:
        return extract_text_from_txt(file_bytes)
    if suffix in [".md"]:
        return extract_text_from_md(file_bytes)
    if suffix in [".json"]:
        return extract_text_from_json(file_bytes)
    if suffix in [".csv"]:
        return extract_text_from_csv(file_bytes)
    if suffix in [".pdf"]:
        return extract_text_from_pdf(file_bytes)
    if suffix in [".docx"]:
        return extract_text_from_docx(file_bytes)

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def ingest_uploaded_files(uploaded_files):
    vectordb = get_vectorstore()

    total_chunks = 0
    processed_files = []

    for uploaded_file in uploaded_files:
        raw_text = extract_text(uploaded_file)

        if not raw_text.strip():
            st.warning(f"No readable text found in {uploaded_file.name}. Skipping.")
            continue

        save_uploaded_file(uploaded_file)

        chunks = chunk_text(raw_text)
        metadatas = [{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]

        vectordb.add_texts(texts=chunks, metadatas=metadatas)
        total_chunks += len(chunks)
        processed_files.append(uploaded_file.name)

    # Persist to disk
    if hasattr(vectordb, "persist"):
        vectordb.persist()

    return processed_files, total_chunks


def build_prompt(question: str, docs):
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Document {i} | Source: {source}]\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    prompt = f"""
You are the Brag & Bev AI Agent.

Answer the user's question using ONLY the provided context when possible.
If the answer is not in the context, say that you do not have enough information in the uploaded documents.
Be clear, direct, and useful.

User question:
{question}

Context:
{context}

Answer:
""".strip()

    return prompt


def answer_question(question: str, k: int = 4):
    vectordb = get_vectorstore()
    llm = get_llm()

    docs = vectordb.similarity_search(question, k=k)

    if not docs:
        return "I couldn't find any relevant documents in the knowledge base yet. Upload files first.", []

    prompt = build_prompt(question, docs)
    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    return answer, docs


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "md", "pdf", "docx", "csv", "json"],
        accept_multiple_files=True,
        help="Supported: TXT, MD, PDF, DOCX, CSV, JSON",
    )

    if st.button("Ingest Uploaded Files", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            with st.spinner("Reading files, chunking text, and building the vector database..."):
                try:
                    processed_files, total_chunks = ingest_uploaded_files(uploaded_files)
                    st.success(
                        f"Ingestion complete. Files processed: {len(processed_files)} | Chunks added: {total_chunks}"
                    )
                    if processed_files:
                        st.write("Processed files:")
                        for f in processed_files:
                            st.write(f"- {f}")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.button("Show Stored Raw Files", use_container_width=True):
        raw_files = sorted([p.name for p in Path(UPLOAD_DIR).glob("*") if p.is_file()])
        if raw_files:
            st.write("Files in data/raw:")
            for f in raw_files:
                st.write(f"- {f}")
        else:
            st.info("No files saved in data/raw yet.")

    if st.button("Reset Vector Database", use_container_width=True):
        try:
            # Clear cache so a fresh vector store is created
            st.cache_resource.clear()

            # Delete persisted Chroma files
            chroma_dir = Path(PERSIST_DIR)
            if chroma_dir.exists():
                for item in chroma_dir.rglob("*"):
                    if item.is_file():
                        item.unlink()
                for item in sorted(chroma_dir.rglob("*"), reverse=True):
                    if item.is_dir():
                        try:
                            item.rmdir()
                        except OSError:
                            pass
                chroma_dir.mkdir(parents=True, exist_ok=True)

            st.success("Vector database reset. You can ingest files again.")
        except Exception as e:
            st.error(f"Reset failed: {e}")


# -----------------------------
# MAIN Q&A
# -----------------------------
question = st.text_input("Ask a question")

if st.button("Run AI"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running AI..."):
            try:
                answer, docs = answer_question(question)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("Retrieved Context"):
                    if docs:
                        for i, doc in enumerate(docs, start=1):
                            st.markdown(f"**Document {i}**")
                            st.write(f"Source: {doc.metadata.get('source', 'unknown')}")
                            st.write(doc.page_content[:1200] + ("..." if len(doc.page_content) > 1200 else ""))
                            st.markdown("---")
                    else:
                        st.write("No documents retrieved.")
            except Exception as e:
                st.error(f"AI run failed: {e}")
"""
Brag & Bev AI System.
Lightweight multi-agent document QA for Render using processed local files.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Brag & Bev AI System",
    page_icon="📄",
    layout="wide",
)


def _smoke_requested() -> bool:
    try:
        query_params = getattr(st, "query_params", None)
        if query_params is None:
            return False
        value = query_params.get("smoke", "")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value).strip().lower() in {"1", "true", "yes"}
    except Exception:
        return False


if _smoke_requested():
    st.success("Smoke test OK — Streamlit is running.")
    st.caption("This mode skips retrieval and Groq so you can isolate frontend/startup issues.")
    st.code(f"Python {sys.version}\nStreamlit {st.__version__}", language="text")
    st.stop()

try:
    from src.agents import AGENTS, DEFAULT_AGENT_KEY, get_agent
    from src.file_processing import (
        PROCESSED_DIR,
        SOURCE_DIR,
        count_supported_source_files,
        get_processed_file_count,
    )
    from src.prompts import build_user_prompt
    from src.retrieval import build_corpus, retrieve_chunks
except Exception as exc:
    st.error(f"Startup import failure: {exc}")
    st.stop()

APP_TITLE = "Brag & Bev AI System"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it in Render Dashboard -> Environment or in a local .env file."
        )

    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError(f"Groq SDK import failed: {exc}") from exc

    return Groq(api_key=api_key)


@st.cache_resource(show_spinner=False)
def load_corpus(processed_dir_str: str):
    return build_corpus(Path(processed_dir_str))


def run_agent(question: str, agent_key: str, top_k: int = TOP_K):
    agent = get_agent(agent_key)

    if not PROCESSED_DIR.exists():
        return "Processed folder `data/raw` is missing. Run `python ingest_local.py` first.", []

    if get_processed_file_count(PROCESSED_DIR) == 0:
        return "No processed files found in `data/raw`. Run `python ingest_local.py` first.", []

    try:
        corpus = load_corpus(str(PROCESSED_DIR.resolve()))
    except Exception as exc:
        return f"Retrieval setup failed: {exc}", []

    if not corpus.documents:
        return "No readable processed files were loaded from `data/raw`.", []

    if not corpus.chunks:
        return "Processed files were found, but no usable chunks were created.", []

    try:
        retrieved = retrieve_chunks(question, corpus, top_k=top_k)
    except Exception as exc:
        return f"Retrieval failure: {exc}", []

    if not retrieved:
        return "No matching context was found for that question.", []

    try:
        client = get_groq_client()
    except Exception as exc:
        return str(exc), retrieved

    user_prompt = build_user_prompt(agent, question, retrieved)

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = completion.choices[0].message.content or "No response content returned."
    except Exception as exc:
        return f"Groq API failure: {exc}", retrieved

    return answer.strip(), retrieved


def render_sidebar(selected_agent_key: str):
    source_exists = SOURCE_DIR.exists()
    processed_exists = PROCESSED_DIR.exists()
    source_count = count_supported_source_files(SOURCE_DIR)
    processed_count = get_processed_file_count(PROCESSED_DIR)
    has_api_key = bool(os.environ.get("GROQ_API_KEY"))

    with st.sidebar:
        st.header("Workspace")
        agent_labels = {key: cfg.label for key, cfg in AGENTS.items()}
        chosen_label = st.selectbox(
            "Choose agent",
            options=list(agent_labels.values()),
            index=list(AGENTS.keys()).index(selected_agent_key),
        )
        selected_key = next(key for key, label in agent_labels.items() if label == chosen_label)

        st.divider()
        st.subheader("Status")
        st.caption(f"Source files: {source_count}")
        st.caption(f"Processed files: {processed_count}")

        if source_exists:
            st.success("`data/source` is available.")
        else:
            st.warning("`data/source` is missing.")

        if processed_exists:
            if processed_count:
                st.success("`data/raw` is ready for retrieval.")
            else:
                st.warning("`data/raw` exists but has no processed files.")
        else:
            st.warning("`data/raw` is missing.")

        if has_api_key:
            st.success("Groq API key detected.")
        else:
            st.error("Missing `GROQ_API_KEY`.")

        st.divider()
        st.subheader("Workflow")
        st.caption("1. Put files in `data/source`.")
        st.caption("2. Run `python ingest_local.py`.")
        st.caption("3. Commit `data/raw`.")
        st.caption("4. Push to GitHub and let Render redeploy.")

    return selected_key


st.title(APP_TITLE)
st.caption(
    "Business-focused multi-agent retrieval over processed local documents. "
    "Use Product, Research, Business, or Communications modes over the same corpus."
)

selected_agent_key = render_sidebar(DEFAULT_AGENT_KEY)
selected_agent = get_agent(selected_agent_key)

st.subheader(selected_agent.label)
st.write(selected_agent.description)

question = st.text_area(
    "Ask a question",
    height=140,
    placeholder="Ask about Dumpster Diver, business planning, partner notes, research, or communications.",
)

if st.button("Run AI", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Loading processed files, ranking chunks, and calling Groq..."):
            answer, retrieved_chunks = run_agent(question.strip(), selected_agent.key)
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved context", expanded=False):
            if retrieved_chunks:
                for index, chunk in enumerate(retrieved_chunks, start=1):
                    st.markdown(
                        f"**{index}. {chunk.source}**  \n"
                        f"Score: `{chunk.score:.4f}`"
                    )
                    st.text(chunk.text[:900] + ("..." if len(chunk.text) > 900 else ""))
                    st.divider()
            else:
                st.write("No chunks retrieved.")

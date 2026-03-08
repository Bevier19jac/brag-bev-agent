"""
Brag and Bev AI Agent — RAG chat over your documents (e.g. dumpster diver).
Runs locally or in the cloud (e.g. Render). Uses Groq for LLM and Chroma for retrieval.
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.retriever import load_vectorstore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Page config
st.set_page_config(
    page_title="Brag and Bev Agent",
    page_icon="🍹",
    layout="centered",
)

# System prompt for the RAG agent
SYSTEM_PROMPT = """You are the Brag and Bev company AI assistant. Answer questions using only the provided context from company documents. If the context doesn't contain enough information, say so. Be concise and helpful. When asked about specific topics (e.g. the "dumpster diver"), use the retrieved context to answer."""

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it in Render Dashboard → Environment, or in a .env file locally.")
        st.stop()
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.2,
        api_key=api_key,
    )

@st.cache_resource
def get_retriever():
    try:
        vectorstore = load_vectorstore()
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Could not load document index: {e}. Run index_docs.py and ensure data/chroma exists (or is created at build time).")
        st.stop()

def get_rag_chain():
    llm = get_llm()
    retriever = get_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)

def main():
    st.title("🍹 Brag and Bev Agent")
    st.caption("Ask about your docs — e.g. the dumpster diver, products, brand, partner notes.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about the dumpster diver or anything in your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chain = get_rag_chain()
                    result = chain.invoke({"input": prompt})
                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err = str(e)
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})

if __name__ == "__main__":
    main()

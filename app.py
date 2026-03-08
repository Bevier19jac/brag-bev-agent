import os
import streamlit as st

st.set_page_config(page_title="Brag & Bev AI Agent", layout="wide")

st.title("Brag & Bev AI Agent")

question = st.text_input("Ask a question")
run_button = st.button("Run AI")

if run_button and question:
    st.write("Running AI...")

    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA

    embeddings = HuggingFaceEmbeddings()

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    result = qa.run(question)

    st.write(result)
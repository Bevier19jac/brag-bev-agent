import streamlit as st

st.set_page_config(page_title="Brag & Bev AI Agent", layout="wide")

st.title("Brag & Bev AI Agent")

question = st.text_input("Ask a question")

if st.button("Run AI"):

    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.llms import Groq

    embeddings = HuggingFaceEmbeddings()

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()

    llm = Groq(model="mixtral-8x7b-32768")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    result = qa.run(question)

    st.write(result)
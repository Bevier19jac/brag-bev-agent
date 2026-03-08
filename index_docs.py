from src.ingest import load_documents, split_documents
from src.retriever import build_vectorstore

print("Loading documents...")
docs = load_documents()

print(f"Loaded {len(docs)} document pages/sections.")

print("Splitting documents into chunks...")
chunks = split_documents(docs)

print(f"Created {len(chunks)} chunks.")

print("Building vector store...")
build_vectorstore(chunks)

print("Done. Documents indexed into Chroma.")
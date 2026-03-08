from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(base_path="data/raw"):
    docs = []
    base = Path(base_path)

    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(str(file_path)).load()
            elif suffix in [".docx", ".doc"]:
                loaded = Docx2txtLoader(str(file_path)).load()
            elif suffix in [".txt", ".md"]:
                loaded = TextLoader(str(file_path), encoding="utf-8").load()
            else:
                continue

            category = file_path.parent.name

            for doc in loaded:
                doc.metadata["source"] = str(file_path)
                doc.metadata["category"] = category

            docs.extend(loaded)

        except Exception as e:
            print(f"Skipped {file_path}: {e}")

    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)
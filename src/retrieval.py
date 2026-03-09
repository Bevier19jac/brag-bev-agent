from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessedDocument:
    source: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class CorpusBundle:
    documents: list[ProcessedDocument]
    chunks: list[RetrievedChunk]
    vectorizer: object
    matrix: object


def read_processed_documents(processed_dir: Path) -> list[ProcessedDocument]:
    if not processed_dir.exists():
        return []

    documents: list[ProcessedDocument] = []
    for path in sorted(processed_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not text:
            continue
        documents.append(ProcessedDocument(source=path.name, text=text))
    return documents


def chunk_text(text: str, source: str, chunk_size: int = 900, overlap: int = 150) -> list[RetrievedChunk]:
    if not text.strip():
        return []

    chunks: list[RetrievedChunk] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end]

        if end < len(normalized):
            last_break = chunk.rfind("\n")
            if last_break > chunk_size // 2:
                chunk = chunk[:last_break]
                end = start + len(chunk)

        chunk = chunk.strip()
        if chunk:
            chunks.append(RetrievedChunk(source=source, text=chunk, score=0.0))

        if end >= len(normalized):
            break

        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks


def build_corpus(processed_dir: Path) -> CorpusBundle:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError(f"scikit-learn is required for retrieval: {exc}") from exc

    documents = read_processed_documents(processed_dir)
    chunks: list[RetrievedChunk] = []
    for document in documents:
        chunks.extend(chunk_text(document.text, document.source))

    if not chunks:
        return CorpusBundle(documents=documents, chunks=[], vectorizer=None, matrix=None)

    chunk_texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1, 2))

    try:
        matrix = vectorizer.fit_transform(chunk_texts)
    except Exception as exc:
        raise RuntimeError(f"TF-IDF build failed: {exc}") from exc

    return CorpusBundle(documents=documents, chunks=chunks, vectorizer=vectorizer, matrix=matrix)


def retrieve_chunks(question: str, corpus: CorpusBundle, top_k: int = 5) -> list[RetrievedChunk]:
    if not question.strip():
        return []
    if not corpus.chunks or corpus.vectorizer is None or corpus.matrix is None:
        return []

    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError(f"scikit-learn is required for retrieval: {exc}") from exc

    try:
        question_vector = corpus.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, corpus.matrix).ravel()
    except Exception as exc:
        raise RuntimeError(f"Similarity search failed: {exc}") from exc

    ranked_indices = scores.argsort()[::-1]
    results: list[RetrievedChunk] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        chunk = corpus.chunks[index]
        results.append(RetrievedChunk(source=chunk.source, text=chunk.text, score=score))
        if len(results) >= top_k:
            break

    return results

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASELINE_DIR = Path("data/baseline")


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


def _iter_txt_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def choose_k(question: str, default_k: int = 8, broad_k: int = 12) -> int:
    """Use broader retrieval for broad/open-ended questions."""
    q = question.lower().strip()
    broad_phrases = (
        "tell me everything you know",
        "tell me everything about",
        "what do you know about",
        "summarize everything you know",
        "summarize everything about",
        "summarize ",
        "explain the project",
        "give me the big picture",
        "overview of",
        "big picture",
    )
    if any(phrase in q for phrase in broad_phrases):
        return broad_k
    return default_k


def read_processed_documents(processed_dir: Path) -> list[ProcessedDocument]:
    documents: list[ProcessedDocument] = []

    for path in _iter_txt_files(processed_dir) + _iter_txt_files(BASELINE_DIR):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
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
    candidates: list[RetrievedChunk] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        chunk = corpus.chunks[index]
        candidates.append(RetrievedChunk(source=chunk.source, text=chunk.text, score=score))
        if len(candidates) >= top_k * 2:
            break

    # Prefer diversity across source files: take one strong chunk per file first, then fill with remainder
    selected: list[RetrievedChunk] = []
    seen_sources: set[str] = set()
    for c in candidates:
        if c.source not in seen_sources:
            selected.append(c)
            seen_sources.add(c.source)
            if len(selected) >= top_k:
                break
    for c in candidates:
        if len(selected) >= top_k:
            break
        if c not in selected:
            selected.append(c)

    return selected

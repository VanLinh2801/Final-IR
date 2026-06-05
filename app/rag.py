from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings


LOGGER = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    stride = chunk_size - chunk_overlap
    while start < len(normalized):
        candidate_end = min(len(normalized), start + chunk_size)
        end = candidate_end
        if candidate_end < len(normalized):
            newline_break = normalized.rfind("\n", start, candidate_end)
            space_break = normalized.rfind(" ", start, candidate_end)
            best_break = max(newline_break, space_break)
            if best_break > start + (chunk_size // 2):
                end = best_break
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start += stride
    return chunks


@dataclass
class RetrievalResult:
    chunks: list[str]
    scores: list[float]


class RagService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = SentenceTransformer(
            str(settings.embedding_model_path),
            local_files_only=True,
        )
        self._lock = Lock()
        self._doc_id: str | None = None
        self._chunks: list[str] = []
        self._embeddings: np.ndarray | None = None
        LOGGER.info("Embedding model loaded from %s", settings.embedding_model_path)

    def ingest(self, text: str, doc_id: str | None) -> tuple[str | None, int]:
        chunks = _chunk_text(
            text,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("Document did not produce any chunks")

        LOGGER.info("Encoding %s chunks for doc_id=%s", len(chunks), doc_id)
        embeddings = self._model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        with self._lock:
            self._doc_id = doc_id
            self._chunks = chunks
            self._embeddings = embeddings

        return self._doc_id, len(chunks)

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        with self._lock:
            embeddings = self._embeddings
            chunks = list(self._chunks)

        if embeddings is None or not chunks:
            raise LookupError("No document has been uploaded yet")

        query_vector = self._model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        scores = embeddings @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]
        selected_chunks = [chunks[index] for index in top_indices]
        selected_scores = [float(scores[index]) for index in top_indices]
        return RetrievalResult(chunks=selected_chunks, scores=selected_scores)

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._embeddings is not None and bool(self._chunks)

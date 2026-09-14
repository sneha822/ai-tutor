"""
Local RAG over ./materials: section-aware chunking -> MiniLM embeddings -> persistent ChromaDB.

Files are re-indexed only when their contents change. Embeddings are always computed here and passed
to Chroma, so Chroma never downloads its own embedding model. Everything works offline once the
MiniLM weights are cached.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

import config

log = logging.getLogger("rag")

ROOT = Path(__file__).resolve().parent.parent
SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass
class Chunk:
    text: str
    source: str
    section: str
    distance: float


def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(errors="ignore")


def _windows(paragraph: str) -> list[str]:
    if len(paragraph) <= config.CHUNK_CHARS:
        return [paragraph]
    step = config.CHUNK_CHARS - config.CHUNK_OVERLAP
    return [paragraph[i:i + config.CHUNK_CHARS] for i in range(0, len(paragraph) - config.CHUNK_OVERLAP, step)]


def split(text: str) -> list[tuple[str, str]]:
    """Split into (section heading, chunk) pairs. Markdown headings start new sections."""
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))

    chunks = []
    for heading, body in sections:
        paragraphs = [w for p in re.split(r"\n\s*\n", body) if p.strip() for w in _windows(p.strip())]
        cur = ""
        for p in paragraphs:
            if cur and len(cur) + len(p) + 2 > config.CHUNK_CHARS:
                chunks.append((heading, cur))
                cur = cur[-config.CHUNK_OVERLAP:] + "\n\n" + p
            else:
                cur = f"{cur}\n\n{p}" if cur else p
        if cur:
            chunks.append((heading, cur))
    return chunks


def _load_embedder():
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(config.EMBED_MODEL, device="cpu", local_files_only=True)
    except Exception:
        log.info("Embedding model not cached; downloading %s", config.EMBED_MODEL)
        return SentenceTransformer(config.EMBED_MODEL, device="cpu")


class Retriever:
    def __init__(self):
        t0 = time.perf_counter()
        self._model = _load_embedder()
        client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR),
                                           settings=Settings(anonymized_telemetry=False))
        self._col = client.get_or_create_collection(
            "materials", metadata={"hnsw:space": "cosine"}, embedding_function=None)
        self.sync()
        log.info("RAG ready in %.1fs (%d chunks)", time.perf_counter() - t0, self._col.count())

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def sync(self) -> None:
        """Index new/changed files in MATERIALS_DIR and drop deleted ones."""
        materials = ROOT / config.MATERIALS_DIR
        materials.mkdir(exist_ok=True)
        files = sorted(p for p in materials.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
        current = {str(p.relative_to(materials)): p for p in files}

        existing: dict[str, str] = {}
        for meta in self._col.get(include=["metadatas"])["metadatas"]:
            existing[meta["source"]] = meta["file_hash"]

        for source in set(existing) - set(current):
            self._col.delete(where={"source": source})
            log.info("Removed %s from index", source)

        for source, path in current.items():
            try:
                file_hash = hashlib.sha1(path.read_bytes()).hexdigest()
                if existing.get(source) == file_hash:
                    continue
                self._col.delete(where={"source": source})
                pieces = split(_load_text(path))
                if not pieces:
                    log.warning("No text extracted from %s", source)
                    continue
                self._col.add(
                    ids=[f"{source}::{i}" for i in range(len(pieces))],
                    documents=[text for _, text in pieces],
                    embeddings=self._embed([f"{heading}\n{text}" for heading, text in pieces]),
                    metadatas=[{"source": source, "section": heading, "file_hash": file_hash}
                               for heading, _ in pieces],
                )
                log.info("Indexed %s (%d chunks)", source, len(pieces))
            except Exception:
                log.exception("Failed to index %s (skipping)", source)

    def retrieve(self, query: str, k: int = config.RAG_TOP_K) -> list[Chunk]:
        """Top-k relevant chunks. Never raises; returns [] on failure."""
        try:
            count = self._col.count()
            if count == 0:
                return []
            t0 = time.perf_counter()
            res = self._col.query(query_embeddings=self._embed([query]), n_results=min(k, count),
                                  include=["documents", "metadatas", "distances"])
            hits = [Chunk(doc, meta["source"], meta.get("section", ""), dist)
                    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
            kept = [c for c in hits if c.distance <= config.RAG_MAX_DISTANCE]
            log.info("rag %.0fms: %d/%d chunks kept (best distance %.2f)",
                     (time.perf_counter() - t0) * 1000, len(kept), len(hits), hits[0].distance if hits else -1)
            return kept
        except Exception:
            log.exception("RAG retrieval failed (continuing without context)")
            return []

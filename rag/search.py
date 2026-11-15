"""Vector (Qdrant + fastembed), BM25, hybrid (reciprocal rank fusion) and reranked search with metadata filters."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / ".cache" / "models"
EMB_CACHE_DIR = ROOT / ".cache" / "embeddings"

MODES = ("bm25", "vector", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class Filters:
    """Metadata filters; None means "no constraint"."""

    doc_set: str | None = None          # "open_api" | "booking_engine"
    updated_after: str | None = None    # ISO date, e.g. "2026-01-01"
    formats: tuple[str, ...] | None = None

    def allows(self, c: dict) -> bool:
        return (
            (self.doc_set is None or c["doc_set"] == self.doc_set)
            and (self.updated_after is None or c["updated_at"] >= self.updated_after)
            and (self.formats is None or c["format"] in self.formats)
        )

    def to_qdrant(self) -> models.Filter | None:
        must = []
        if self.doc_set:
            must.append(models.FieldCondition(key="doc_set", match=models.MatchValue(value=self.doc_set)))
        if self.updated_after:
            # Dates are stored as ordinal days so Qdrant can range-filter them.
            must.append(models.FieldCondition(key="updated_ord", range=models.Range(gte=_ord(self.updated_after))))
        if self.formats:
            must.append(models.FieldCondition(key="format", match=models.MatchAny(any=list(self.formats))))
        return models.Filter(must=must) if must else None


def _ord(iso: str) -> int:
    from datetime import date

    return date.fromisoformat(iso[:10]).toordinal()


def load_chunks(strategy: str) -> list[dict]:
    path = PROCESSED / f"chunks_{strategy}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `make ingest` first")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def index_text(c: dict) -> str:
    """Contextual header + chunk: the title and section path disambiguate generic chunks ("Step by step:")."""
    return f"{c['heading_path']}\n{c['text']}"


_TOKEN = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Keeps API identifiers whole (reservation.created.v2, /reservations-v3) and also indexes their parts."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        parts = re.split(r"[._/-]", tok)
        if len(parts) > 1:
            out.extend(p for p in parts if p)
    return out


class EmbeddingCache:
    """sha256(model + text) -> vector, persisted per model; a re-run embeds only new or changed chunks."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.path = EMB_CACHE_DIR / (re.sub(r"[^\w.-]", "_", model_name) + ".npz")
        self._model: TextEmbedding | None = None
        self.store: dict[str, np.ndarray] = {}
        if self.path.exists():
            data = np.load(self.path)
            self.store = {k: data[k] for k in data.files}
        self.hits = self.misses = 0

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(self.model_name, cache_dir=str(MODELS_DIR))
        return self._model

    def _key(self, text: str) -> str:
        return hashlib.sha256((self.model_name + "\n" + text).encode()).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.store]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            vectors = list(self.model.embed([t for _, t in missing], batch_size=32))
            for (k, _), v in zip(missing, vectors):
                self.store[k] = np.asarray(v, dtype=np.float32)
            EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, **self.store)
        return np.stack([self.store[k] for k in keys])

    def embed_query(self, query: str) -> np.ndarray:
        # Queries are not cached on disk: they are cheap and change every run.
        return np.asarray(next(iter(self.model.query_embed([query]))), dtype=np.float32)


class Retriever:
    def __init__(self, chunks: list[dict], embed_model: str, rerank_model: str, candidates: int = 20, rrf_k: int = 60):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.emb = EmbeddingCache(embed_model)
        self.rerank_model = rerank_model
        self._reranker: TextCrossEncoder | None = None

        vectors = self.emb.embed([index_text(c) for c in chunks])
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            "chunks", vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE)
        )
        self.qdrant.upsert(
            "chunks",
            points=[
                models.PointStruct(
                    id=i,
                    vector=v.tolist(),
                    payload={**{k: c[k] for k in ("chunk_id", "doc_id", "doc_set", "format", "updated_at")}, "updated_ord": _ord(c["updated_at"])},
                )
                for i, (c, v) in enumerate(zip(chunks, vectors))
            ],
        )
        self.bm25 = BM25Okapi([tokenize(index_text(c)) for c in chunks])

    @property
    def reranker(self) -> TextCrossEncoder:
        if self._reranker is None:
            self._reranker = TextCrossEncoder(self.rerank_model, cache_dir=str(MODELS_DIR))
        return self._reranker

    def vector(self, query: str, filters: Filters, k: int) -> list[tuple[dict, float]]:
        q = self.emb.embed_query(query)
        hits = self.qdrant.query_points("chunks", query=q.tolist(), limit=k, query_filter=filters.to_qdrant()).points
        return [(self.chunks[h.id], float(h.score)) for h in hits]

    def lexical(self, query: str, filters: Filters, k: int) -> list[tuple[dict, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if filters.allows(self.chunks[i]):
                out.append((self.chunks[i], float(scores[i])))
                if len(out) == k:
                    break
        return out

    def hybrid(self, query: str, filters: Filters, k: int) -> list[tuple[dict, float]]:
        fused: dict[str, float] = {}
        for ranked in (self.vector(query, filters, self.candidates), self.lexical(query, filters, self.candidates)):
            for rank, (c, _) in enumerate(ranked):
                fused[c["chunk_id"]] = fused.get(c["chunk_id"], 0.0) + 1.0 / (self.rrf_k + rank + 1)
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(self.by_id[cid], s) for cid, s in best]

    def rerank(self, query: str, candidates: list[tuple[dict, float]], k: int) -> list[tuple[dict, float]]:
        if not candidates:
            return []
        scores = list(self.reranker.rerank(query, [index_text(c) for c, _ in candidates]))
        ranked = sorted(zip((c for c, _ in candidates), scores), key=lambda x: -x[1])
        return [(c, float(s)) for c, s in ranked[:k]]

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 5, filters: Filters | None = None) -> list[tuple[dict, float]]:
        filters = filters or Filters()
        if mode == "bm25":
            return self.lexical(query, filters, k)
        if mode == "vector":
            return self.vector(query, filters, k)
        if mode == "hybrid":
            return self.hybrid(query, filters, k)
        if mode == "hybrid_rerank":
            return self.rerank(query, self.hybrid(query, filters, self.candidates), k)
        raise ValueError(f"unknown mode {mode!r}")

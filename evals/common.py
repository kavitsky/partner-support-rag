"""Test set loading and the relevance rule shared by the retrieval and answer evals."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "evals" / "testset.jsonl"
RESULTS = ROOT / "results"


def load_testset(path: Path = TESTSET) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: dict, item: dict) -> bool:
    """A chunk is relevant if it comes from an evidence doc and contains that doc's evidence quote.

    Labels are (doc_id, quote) pairs rather than chunk ids, so they stay valid when the chunking strategy changes.
    """
    text = norm(chunk["text"])
    return any(chunk["doc_id"] == ev["doc_id"] and norm(ev["quote"]) in text for ev in item["evidence"])


def matches(text: str, fact: str) -> bool:
    """`fact` may list alternatives separated by '|'; any one present (case-insensitive) counts."""
    t = norm(text)
    return any(norm(alt) in t for alt in fact.split("|"))

"""Grounded answers: retrieve, pack sources into a token budget, answer with per-claim citations or abstain."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from rag.ingest import n_tokens, _enc
from rag.llm import LLM, Completion
from rag.search import Filters, Retriever

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"

NOT_IN_DOCS = "This is not covered in the documentation I have access to."
DOC_SET_LABEL = {"open_api": "Open API", "booking_engine": "Booking Engine API"}


class Claim(BaseModel):
    text: str
    sources: list[str]


class Answer(BaseModel):
    status: Literal["answered", "not_in_docs"]
    claims: list[Claim]
    missing: str


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "not_in_docs"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}}},
                "required": ["text", "sources"],
                "additionalProperties": False,
            },
        },
        "missing": {"type": "string"},
    },
    "required": ["status", "claims", "missing"],
    "additionalProperties": False,
}

REPAIR = "Your previous answer was invalid: {error}\nReturn the corrected JSON object only."


@dataclass(frozen=True)
class Prompt:
    kind: str
    version: str
    system: str
    user: str

    @property
    def sha(self) -> str:
        return hashlib.sha256((self.system + self.user).encode()).hexdigest()[:12]


def load_prompt(kind: str, version: str) -> Prompt:
    d = PROMPTS / kind / version
    return Prompt(kind, version, (d / "system.txt").read_text(encoding="utf-8"), (d / "user.txt").read_text(encoding="utf-8"))


def pack_context(hits: list[tuple[dict, float]], budget: int) -> list[dict]:
    """Best-ranked chunks first until the token budget is spent; the first chunk is truncated rather than dropped."""
    sources, used = [], 0
    for chunk, score in hits:
        header = (
            f"{chunk['title']} › {chunk['section']} | {DOC_SET_LABEL[chunk['doc_set']]} | updated {chunk['updated_at']}"
        )
        text = chunk["text"]
        cost = n_tokens(header) + n_tokens(text) + 4
        if used + cost > budget:
            if sources:
                continue
            text = _enc.decode(_enc.encode(text)[: budget - n_tokens(header) - 4])
            cost = budget
        sid = f"S{len(sources) + 1}"
        sources.append({"sid": sid, "header": header, "text": text, "chunk": chunk, "score": score})
        used += cost
    return sources


def render_context(sources: list[dict]) -> str:
    return "\n\n".join(f"[{s['sid']}] {s['header']}\n{s['text']}" for s in sources)


def _validate(content: str, valid_ids: set[str]) -> tuple[Answer | None, str | None]:
    try:
        a = Answer.model_validate_json(content)
    except ValidationError as e:
        return None, str(e)
    if a.status == "answered":
        if not a.claims:
            return None, 'status is "answered" but there are no claims'
        bad = sorted({s for c in a.claims for s in c.sources} - valid_ids)
        if bad:
            return None, f"claims cite unknown sources {bad}; use only {sorted(valid_ids)}"
        if any(not c.sources for c in a.claims):
            return None, "every claim must cite at least one source"
    return a, None


def render_answer(a: Answer, sources: list[dict]) -> str:
    if a.status == "not_in_docs":
        return NOT_IN_DOCS + (f" ({a.missing})" if a.missing else "")
    by_sid = {s["sid"]: s for s in sources}
    lines = []
    for c in a.claims:
        cites = "; ".join(f"{s}: {by_sid[s]['chunk']['title']} › {by_sid[s]['chunk']['section']}" for s in c.sources)
        lines.append(f"- {c.text} [{cites}]")
    if a.missing:
        lines.append(f"- Not covered by the docs: {a.missing}")
    return "\n".join(lines)


class RAG:
    def __init__(self, cfg: dict, retriever: Retriever, llm: LLM):
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm
        self.prompt = load_prompt("answer", cfg["rag"]["prompt_version"])

    def answer(self, question: str, filters: Filters | None = None) -> dict:
        rc, rag = self.cfg["retrieval"], self.cfg["rag"]
        hits = self.retriever.search(question, rc["mode"], rc["k"], filters)
        record = {
            "question": question,
            "filters": (filters or Filters()).__dict__,
            "model": self.llm.model,
            "prompt_version": self.prompt.version,
            "prompt_sha": self.prompt.sha,
            "retrieved": [{"chunk_id": c["chunk_id"], "score": round(s, 3)} for c, s in hits],
            "gate": None,
            "attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "latency_s": 0.0,
            "cached": False,
            "error": None,
        }
        top = hits[0][1] if hits else float("-inf")
        if rc["mode"] == "hybrid_rerank" and top < rag["min_rerank_score"]:
            record["gate"] = f"best rerank score {top:.2f} < {rag['min_rerank_score']}"
            a = Answer(status="not_in_docs", claims=[], missing="no relevant passage was retrieved")
            return {**record, "sources": [], "context_tokens": 0, **self._finish(a, [])}

        sources = pack_context(hits, rag["context_token_budget"])
        messages = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": self.prompt.user.replace("{context}", render_context(sources)).replace("{question}", question)},
        ]
        valid = {s["sid"] for s in sources}
        completions: list[Completion] = []
        a, error = None, None
        try:
            comp = self.llm.complete(messages, ANSWER_SCHEMA, "grounded_answer")
            completions.append(comp)
            a, error = _validate(comp.content, valid)
            if a is None:
                messages += [{"role": "assistant", "content": comp.content}, {"role": "user", "content": REPAIR.format(error=error)}]
                comp = self.llm.complete(messages, ANSWER_SCHEMA, "grounded_answer")
                completions.append(comp)
                a, error = _validate(comp.content, valid)
        except Exception as e:  # provider gave up after its retries
            error = f"{type(e).__name__}: {e}"
        if a is None:
            record["error"] = error
            a = Answer(status="not_in_docs", claims=[], missing="the answer could not be generated")

        record.update(
            attempts=len(completions),
            input_tokens=sum(c.input_tokens for c in completions),
            output_tokens=sum(c.output_tokens for c in completions),
            cost_usd=sum(self.llm.cost(c) for c in completions),
            latency_s=round(sum(c.latency_s for c in completions), 3),
            cached=bool(completions) and all(c.cached for c in completions),
        )
        src_out = [
            {
                "sid": s["sid"],
                "chunk_id": s["chunk"]["chunk_id"],
                "doc_id": s["chunk"]["doc_id"],
                "title": s["chunk"]["title"],
                "section": s["chunk"]["section"],
                "url": s["chunk"]["url"],
                "updated_at": s["chunk"]["updated_at"],
                "score": round(s["score"], 3),
                "text": s["text"],
            }
            for s in sources
        ]
        context_tokens = n_tokens(render_context(sources))
        return {**record, "sources": src_out, "context_tokens": context_tokens, **self._finish(a, src_out)}

    @staticmethod
    def _finish(a: Answer, sources: list[dict]) -> dict:
        by_sid = {s["sid"]: s for s in sources}
        rendered_sources = [{"sid": s["sid"], "chunk": {"title": s["title"], "section": s["section"]}} for s in sources]
        return {
            "status": a.status,
            "claims": [c.model_dump() for c in a.claims],
            "missing": a.missing,
            "cited_docs": sorted({by_sid[s]["doc_id"] for c in a.claims for s in c.sources if s in by_sid}),
            "answer": render_answer(a, rendered_sources),
        }

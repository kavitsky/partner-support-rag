"""Retrieval quality on the answerable test questions: recall@k and MRR@10 across search configurations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from evals.common import RESULTS, is_relevant, load_testset
from rag.search import Filters, Retriever, load_chunks


@dataclass(frozen=True)
class Config:
    name: str
    strategy: str
    model: str
    mode: str
    use_surface_filter: bool = False


def configs(cfg: dict) -> list[Config]:
    main_model = cfg["embedding"]["model"]
    alt = [m for m in cfg["embedding"]["compare_models"] if m != main_model]
    out = []
    for strategy in ("structure", "fixed"):
        for mode in ("bm25", "vector", "hybrid", "hybrid_rerank"):
            out.append(Config(f"{strategy} / {mode}", strategy, main_model, mode))
    out.append(Config("structure / hybrid_rerank + surface filter", "structure", main_model, "hybrid_rerank", True))
    for m in alt:
        short = m.split("/")[-1]
        out.append(Config(f"structure / vector ({short})", "structure", m, "vector"))
        out.append(Config(f"structure / hybrid_rerank ({short})", "structure", m, "hybrid_rerank"))
    return out


def evaluate(retriever: Retriever, items: list[dict], c: Config, k: int) -> dict:
    per_q, t0 = [], time.perf_counter()
    for it in items:
        filters = Filters(doc_set=it.get("surface")) if c.use_surface_filter else Filters()
        hits = retriever.search(it["question"], c.mode, 10, filters)
        rel = [is_relevant(ch, it) for ch, _ in hits]
        first = next((i for i, r in enumerate(rel) if r), None)
        docs_top_k = {ch["doc_id"] for ch, _ in hits[:k]}
        per_q.append(
            {
                "id": it["id"],
                "type": it["type"],
                "hit_at_k": any(rel[:k]),
                "rr": 1.0 / (first + 1) if first is not None else 0.0,
                "first_relevant_rank": None if first is None else first + 1,
                "doc_recall_at_k": len(docs_top_k & set(it["expected_docs"])) / len(it["expected_docs"]),
                "top_k": [ch["chunk_id"] for ch, _ in hits[:k]],
            }
        )
    n = len(per_q)
    return {
        "config": c.__dict__,
        f"recall@{k}": round(sum(q["hit_at_k"] for q in per_q) / n, 3),
        "mrr@10": round(sum(q["rr"] for q in per_q) / n, 3),
        f"doc_recall@{k}": round(sum(q["doc_recall_at_k"] for q in per_q) / n, 3),
        "ms_per_query": round((time.perf_counter() - t0) * 1000 / n, 1),
        "misses": [q["id"] for q in per_q if not q["hit_at_k"]],
        "per_question": per_q,
    }


def production_config(cfg: dict) -> Config:
    return Config("production", cfg["chunking"]["strategy"], cfg["embedding"]["model"], cfg["retrieval"]["mode"], True)


def check(cfg: dict) -> tuple[dict, list[str]]:
    """Production configuration only, against thresholds (used in CI)."""
    k = cfg["retrieval"]["k"]
    items = [it for it in load_testset() if it["evidence"]]
    c = production_config(cfg)
    r = evaluate(Retriever(load_chunks(c.strategy), c.model, cfg["retrieval"]["rerank_model"],
                           cfg["retrieval"]["candidates"], cfg["retrieval"]["rrf_k"]), items, c, k)
    t = cfg["eval"]["retrieval_thresholds"]
    failures = [f"{m} = {r[m]} < {v}" for m, v in ((f"recall@{k}", t["recall_at_k"]), ("mrr@10", t["mrr_at_10"])) if r[m] < v]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_check.json").write_text(json.dumps({**r, "thresholds": t, "failures": failures}, indent=2) + "\n")
    print(f"  production retrieval: recall@{k}={r[f'recall@{k}']:.3f}  mrr@10={r['mrr@10']:.3f}  misses={r['misses']}")
    return r, failures


def run(cfg: dict) -> list[dict]:
    k = cfg["retrieval"]["k"]
    items = [it for it in load_testset() if it["evidence"]]
    retrievers: dict[tuple[str, str], Retriever] = {}
    results = []
    for c in configs(cfg):
        key = (c.strategy, c.model)
        if key not in retrievers:
            retrievers[key] = Retriever(
                load_chunks(c.strategy), c.model, cfg["retrieval"]["rerank_model"],
                cfg["retrieval"]["candidates"], cfg["retrieval"]["rrf_k"],
            )
        r = evaluate(retrievers[key], items, c, k)
        results.append(r)
        print(f"  {c.name:<55} recall@{k}={r[f'recall@{k}']:.3f}  mrr@10={r['mrr@10']:.3f}  misses={r['misses']}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps({"questions": len(items), "k": k, "results": results}, indent=2) + "\n")
    lines = [
        f"| Configuration | recall@{k} | MRR@10 | doc recall@{k} | ms/query |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']['name']} | {r[f'recall@{k}']:.3f} | {r['mrr@10']:.3f} | {r[f'doc_recall@{k}']:.3f} | {r['ms_per_query']} |"
        )
    (RESULTS / "retrieval.md").write_text(f"Retrieval on {len(items)} answerable questions.\n\n" + "\n".join(lines) + "\n")
    return results

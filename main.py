"""partner-support-rag: grounded answers over Guesty's public API docs, with retrieval and answer evals."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_config() -> dict:
    return tomllib.loads((ROOT / "config.toml").read_text())


def cmd_fetch(args, cfg):
    from rag.fetch import fetch

    print(json.dumps(fetch(force=args.force), indent=2))


def cmd_ingest(args, cfg):
    from rag.ingest import run

    report = run(cfg)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ingest_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    q = report["chunk_quality"]
    cols = ["chunks", "tokens_mean", "tokens_p95", "tokens_min", "clean_start_pct", "clean_end_pct", "cut_table_or_code_pct"]
    lines = ["| Strategy | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for s, v in q.items():
        lines.append(f"| {s} | " + " | ".join(str(v[c]) for c in cols) + " |")
    lines += ["", "By format (structure strategy):", "", "| Format | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for fmt, v in q["structure"]["by_format"].items():
        lines.append(f"| {fmt} | " + " | ".join(str(v[c]) for c in cols) + " |")
    (RESULTS / "chunking.md").write_text("\n".join(lines) + "\n")
    print(f"{report['documents']} documents ({report['by_format']}), kept {report['kept_documents']}")
    print(f"exact duplicates: {report['exact_duplicates'] or 'none'}; near-duplicates: {report['near_duplicates'] or 'none'}")
    if report["changed_since_manifest"]:
        print(f"changed since the pinned manifest: {report['changed_since_manifest']}")
    print("\n".join(lines[: 2 + len(q)]))


def _retriever(cfg):
    from rag.search import Retriever, load_chunks

    r = cfg["retrieval"]
    return Retriever(load_chunks(cfg["chunking"]["strategy"]), cfg["embedding"]["model"], r["rerank_model"], r["candidates"], r["rrf_k"])


def _filters(args):
    from rag.search import Filters

    return Filters(
        doc_set=getattr(args, "doc_set", None),
        updated_after=getattr(args, "updated_after", None),
        formats=tuple(args.format) if getattr(args, "format", None) else None,
    )


def cmd_index(args, cfg):
    r = _retriever(cfg)
    print(f"indexed {len(r.chunks)} chunks with {cfg['embedding']['model']} "
          f"(embedding cache: {r.emb.hits} reused, {r.emb.misses} computed)")


def cmd_search(args, cfg):
    r = _retriever(cfg)
    for c, s in r.search(args.query, args.mode or cfg["retrieval"]["mode"], args.k, _filters(args)):
        print(f"{s:8.3f}  {c['chunk_id']:<45} {c['heading_path']}  ({c['format']}, {c['updated_at']})")


def cmd_ask(args, cfg):
    from rag.answer import RAG
    from rag.llm import LLM

    rc = cfg["rag"]
    rag = RAG(cfg, _retriever(cfg), LLM(rc["model"], (rc["price_input"], rc["price_output"]), cfg["openrouter"], rc["temperature"]))
    rec = rag.answer(args.question, _filters(args))
    print(rec["answer"])
    print("\nSources:")
    for s in rec["sources"]:
        print(f"  [{s['sid']}] {s['title']} › {s['section']}  {s['url']}")
    print(f"\n{rec['input_tokens']} in / {rec['output_tokens']} out tokens, ${rec['cost_usd']:.5f}, prompt {rec['prompt_version']}"
          + (f", gate: {rec['gate']}" if rec["gate"] else ""))


def cmd_eval_retrieval(args, cfg):
    from evals import retrieval_eval

    if args.check:
        _, failures = retrieval_eval.check(cfg)
        _exit(failures)
    else:
        retrieval_eval.run(cfg)
        print(f"\nwrote {RESULTS / 'retrieval.md'}")


def cmd_eval(args, cfg):
    from evals import answer_eval

    if args.prompt:
        cfg["rag"]["prompt_version"] = args.prompt
    summary, failures = answer_eval.run(cfg, limit=args.limit)
    print("\n" + answer_eval.to_markdown(summary))
    _exit(failures)


def cmd_grade(args, cfg):
    from evals.grade import run

    run(args.versions, args.grader)


def cmd_calibrate(args, cfg):
    from evals.calibrate import run

    run()
    print((RESULTS / "judge_calibration.md").read_text())


def _exit(failures: list[str]) -> None:
    if failures:
        print("FAILED thresholds:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("All thresholds passed.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch", help="download the corpus listed in data/sources.toml")
    s.add_argument("--force", action="store_true", help="re-download files that already exist")
    s.set_defaults(func=cmd_fetch)

    sub.add_parser("ingest", help="extract, clean, dedupe and chunk; writes data/processed/ and the manifest").set_defaults(func=cmd_ingest)
    sub.add_parser("index", help="embed chunks (cached) and build the search index").set_defaults(func=cmd_index)

    def add_filters(sp):
        sp.add_argument("--doc-set", choices=["open_api", "booking_engine"])
        sp.add_argument("--updated-after", metavar="YYYY-MM-DD")
        sp.add_argument("--format", nargs="+", choices=["html", "pdf", "docx"])

    s = sub.add_parser("search", help="show retrieved chunks for a query")
    s.add_argument("query")
    s.add_argument("--mode", choices=["bm25", "vector", "hybrid", "hybrid_rerank"])
    s.add_argument("--k", type=int, default=5)
    add_filters(s)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("ask", help="answer a question with citations")
    s.add_argument("question")
    add_filters(s)
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("eval-retrieval", help="recall@k / MRR across retrieval configurations")
    s.add_argument("--check", action="store_true", help="production config only, exit 1 below thresholds")
    s.set_defaults(func=cmd_eval_retrieval)

    s = sub.add_parser("eval", help="answer evals; exit 1 below thresholds")
    s.add_argument("--limit", type=int)
    s.add_argument("--prompt", help="answer prompt version (default: config.toml)")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("grade", help="blind human grading for judge calibration")
    s.add_argument("--versions", nargs="+", default=["v2", "v1"])
    s.add_argument("--grader", default="human")
    s.set_defaults(func=cmd_grade)

    sub.add_parser("calibrate", help="judge vs human agreement").set_defaults(func=cmd_calibrate)

    args = p.parse_args()
    args.func(args, load_config())


if __name__ == "__main__":
    main()

"""End-to-end answer evals: deterministic checks, LLM-judge correctness, faithfulness, thresholds."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel

from evals.common import RESULTS, load_testset, matches
from rag.answer import RAG, load_prompt
from rag.llm import LLM
from rag.search import Filters, Retriever, load_chunks

VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": list(VERDICT_SCORE)},
    },
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}
FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["supported", "not_supported"]},
                    "reason": {"type": "string"},
                },
                "required": ["index", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


class Correctness(BaseModel):
    reasoning: str
    verdict: Literal["correct", "partially_correct", "incorrect"]


class Judge:
    def __init__(self, cfg: dict):
        j = cfg["judge"]
        self.llm = LLM(j["model"], (j["price_input"], j["price_output"]), cfg["openrouter"])
        self.correctness_prompt = load_prompt("judge/correctness", j["correctness_prompt"])
        self.faithfulness_prompt = load_prompt("judge/faithfulness", j["faithfulness_prompt"])
        self.cost = 0.0

    def _ask(self, prompt, fields: dict, schema: dict, name: str) -> dict:
        user = prompt.user
        for k, v in fields.items():
            user = user.replace("{" + k + "}", v)
        comp = self.llm.complete([{"role": "system", "content": prompt.system}, {"role": "user", "content": user}], schema, name)
        self.cost += self.llm.cost(comp)
        return json.loads(comp.content)

    def correctness(self, item: dict, answer: str) -> dict:
        out = self._ask(
            self.correctness_prompt,
            {"question": item["question"], "expected_answer": item["expected_answer"], "answer": answer},
            CORRECTNESS_SCHEMA,
            "correctness",
        )
        return Correctness(**out).model_dump()

    def faithfulness(self, rec: dict) -> dict | None:
        if not rec["claims"]:
            return None
        by_sid = {s["sid"]: s for s in rec["sources"]}
        cited = sorted({s for c in rec["claims"] for s in c["sources"]}, key=lambda s: int(s[1:]))
        sources = "\n\n".join(f"[{s}] {by_sid[s]['title']} › {by_sid[s]['section']}\n{by_sid[s]['text']}" for s in cited if s in by_sid)
        claims = "\n".join(f"{i}. {c['text']} (cites {', '.join(c['sources'])})" for i, c in enumerate(rec["claims"], 1))
        out = self._ask(self.faithfulness_prompt, {"sources": sources, "claims": claims}, FAITHFULNESS_SCHEMA, "faithfulness")
        verdicts = out["claims"]
        supported = sum(v["verdict"] == "supported" for v in verdicts)
        return {"score": supported / max(len(rec["claims"]), 1), "verdicts": verdicts}


def deterministic(item: dict, rec: dict) -> dict:
    should_abstain = item["type"] == "unanswerable"
    abstained = rec["status"] == "not_in_docs"
    text = " ".join(c["text"] for c in rec["claims"])
    out = {
        "should_abstain": should_abstain,
        "abstained": abstained,
        "abstention_correct": abstained == should_abstain,
        "forbidden_present": [f for f in item["must_not_contain"] if matches(text, f)],
    }
    if not should_abstain:
        out["source_hit"] = bool(set(rec["cited_docs"]) & set(item["expected_docs"]))
        facts = item["key_facts"]
        out["key_facts_missing"] = [f for f in facts if not matches(text, f)]
        out["key_fact_recall"] = 1 - len(out["key_facts_missing"]) / len(facts) if facts else 1.0
    if rec["claims"]:
        valid = {s["sid"] for s in rec["sources"]}
        out["citation_validity"] = sum(bool(c["sources"]) and set(c["sources"]) <= valid for c in rec["claims"]) / len(rec["claims"])
    return out


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def aggregate(rows: list[dict]) -> dict:
    answerable = [r for r in rows if not r["checks"]["should_abstain"]]
    faith = [r["faithfulness"]["score"] for r in rows if r.get("faithfulness")]
    return {
        "n": len(rows),
        "abstention_accuracy": _mean([r["checks"]["abstention_correct"] for r in rows]),
        "source_hit": _mean([r["checks"]["source_hit"] for r in answerable]),
        "key_fact_recall": _mean([r["checks"]["key_fact_recall"] for r in answerable]),
        "forbidden_rate": _mean([bool(r["checks"]["forbidden_present"]) for r in rows]),
        "citation_validity": _mean([r["checks"]["citation_validity"] for r in rows if "citation_validity" in r["checks"]]),
        "judge_correctness": _mean([VERDICT_SCORE[r["judge"]["verdict"]] for r in rows]),
        "faithfulness": _mean(faith),
    }


def run(cfg: dict, limit: int | None = None) -> tuple[dict, list[str]]:
    items = load_testset()[:limit] if limit else load_testset()
    retriever = Retriever(
        load_chunks(cfg["chunking"]["strategy"]), cfg["embedding"]["model"], cfg["retrieval"]["rerank_model"],
        cfg["retrieval"]["candidates"], cfg["retrieval"]["rrf_k"],
    )
    rc = cfg["rag"]
    rag = RAG(cfg, retriever, LLM(rc["model"], (rc["price_input"], rc["price_output"]), cfg["openrouter"], rc["temperature"]))
    judge = Judge(cfg)

    rows = []
    for it in items:
        rec = rag.answer(it["question"], Filters(doc_set=it.get("surface")))
        row = {"id": it["id"], "type": it["type"], **rec}
        row["checks"] = deterministic(it, rec)
        row["judge"] = judge.correctness(it, rec["answer"])
        row["faithfulness"] = judge.faithfulness(rec)
        rows.append(row)
        flag = "ok " if row["judge"]["verdict"] == "correct" else "!! "
        print(f"  {flag}{it['id']} {it['type']:<14} {rec['status']:<12} judge={row['judge']['verdict']:<17} {it['question'][:60]}")

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    n_llm = sum(1 for r in rows if r["attempts"])
    answer_cost = sum(r["cost_usd"] for r in rows)
    summary = {
        "answer_model": rc["model"],
        "answer_prompt": f"{rag.prompt.version} ({rag.prompt.sha})",
        "judge_model": cfg["judge"]["model"],
        "retrieval": f"{cfg['chunking']['strategy']} / {cfg['retrieval']['mode']} / {cfg['embedding']['model']}",
        "overall": aggregate(rows),
        "by_type": {t: aggregate(rs) for t, rs in sorted(by_type.items())},
        "cost": {
            "answer_usd_total": round(answer_cost, 5),
            "judge_usd_total": round(judge.cost, 5),
            "answer_usd_per_1000_questions": round(answer_cost / max(len(rows), 1) * 1000, 3),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in rows) / max(n_llm, 1)),
            "mean_output_tokens": round(sum(r["output_tokens"] for r in rows) / max(n_llm, 1)),
            "mean_context_tokens": round(sum(r["context_tokens"] for r in rows) / max(len(rows), 1)),
            "all_cached": all(r["cached"] for r in rows if r["attempts"]),
        },
        "gated": [r["id"] for r in rows if r["gate"]],
        "errors": {r["id"]: r["error"] for r in rows if r["error"]},
    }
    failures = [
        f"{m} = {summary['overall'][m]} < {t}"
        for m, t in cfg["eval"]["thresholds"].items()
        if summary["overall"].get(m) is not None and summary["overall"][m] < t
    ]
    summary["thresholds"] = cfg["eval"]["thresholds"]
    summary["failures"] = failures

    RESULTS.mkdir(exist_ok=True)
    tag = rag.prompt.version
    with (RESULTS / f"answers_{tag}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            # Source text stays out of the repo: only ids, titles and sections are kept.
            slim = {**r, "sources": [{k: v for k, v in s.items() if k != "text"} for s in r["sources"]]}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
    (RESULTS / f"eval_summary_{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (RESULTS / f"eval_{tag}.md").write_text(to_markdown(summary))
    return summary, failures


METRICS = ["abstention_accuracy", "source_hit", "key_fact_recall", "citation_validity", "judge_correctness", "faithfulness", "forbidden_rate"]


def _fmt(v) -> str:
    return "–" if v is None else f"{v:.3f}"


def to_markdown(s: dict) -> str:
    lines = [
        f"Answer model `{s['answer_model']}`, prompt {s['answer_prompt']}; judge `{s['judge_model']}`; retrieval {s['retrieval']}.",
        "",
        "| Slice | n | " + " | ".join(METRICS) + " |",
        "|---|---|" + "---|" * len(METRICS),
        "| **overall** | " + str(s["overall"]["n"]) + " | " + " | ".join(_fmt(s["overall"][m]) for m in METRICS) + " |",
    ]
    for t, agg in s["by_type"].items():
        lines.append(f"| {t} | {agg['n']} | " + " | ".join(_fmt(agg[m]) for m in METRICS) + " |")
    c = s["cost"]
    lines += [
        "",
        f"Cost: answers ${c['answer_usd_total']:.4f}, judge ${c['judge_usd_total']:.4f}; "
        f"**${c['answer_usd_per_1000_questions']:.2f} per 1,000 questions** (answering only); "
        f"mean {c['mean_input_tokens']} input / {c['mean_output_tokens']} output tokens.",
        "",
        "Thresholds: " + ", ".join(f"{m} ≥ {t}" for m, t in s["thresholds"].items()) + ".",
        "Result: " + ("**FAIL** – " + "; ".join(s["failures"]) if s["failures"] else "**PASS**"),
    ]
    return "\n".join(lines) + "\n"

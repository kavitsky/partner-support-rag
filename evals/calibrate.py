"""Agreement between the LLM judge and human grades on the same answers."""

from __future__ import annotations

import json
from collections import Counter

from evals.common import RESULTS
from evals.grade import answer_sha, load_grades

LABELS = ["correct", "partially_correct", "incorrect"]


def cohen_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    ha, ja = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ha[l] * ja[l] for l in labels) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def judged_answers() -> dict[tuple[str, str], dict]:
    out = {}
    for path in sorted(RESULTS.glob("answers_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            out[(r["id"], answer_sha(r["answer"]))] = r
    return out


def run(min_grades: int = 10) -> dict:
    answers = judged_answers()
    pairs, rows, stale = [], [], []
    for g in load_grades():
        r = answers.get((g["id"], g["answer_sha"]))
        if r is None:
            stale.append(g["id"])  # the graded answer is no longer in results/ (prompt or model changed)
            continue
        pairs.append((g["grade"], r["judge"]["verdict"]))
        rows.append({"id": g["id"], "prompt": g["prompt_version"], "human": g["grade"], "judge": r["judge"]["verdict"],
                     "judge_reasoning": r["judge"]["reasoning"], "note": g["note"]})
    if len(pairs) < min_grades:
        raise SystemExit(f"only {len(pairs)} human grades match current answers; grade at least {min_grades} with `make grade`")

    binary = [(h == "correct", j == "correct") for h, j in pairs]
    report = {
        "n": len(pairs),
        "exact_agreement": round(sum(h == j for h, j in pairs) / len(pairs), 3),
        "binary_agreement": round(sum(h == j for h, j in binary) / len(binary), 3),
        "cohen_kappa": round(cohen_kappa(pairs, LABELS), 3),
        "confusion": {h: {j: sum(1 for a, b in pairs if a == h and b == j) for j in LABELS} for h in LABELS},
        "disagreements": [r for r in rows if r["human"] != r["judge"]],
        "stale_grades": stale,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "judge_calibration.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        f"Judge vs human on {report['n']} answers: exact agreement **{report['exact_agreement']:.0%}**, "
        f"binary (correct vs not) **{report['binary_agreement']:.0%}**, Cohen's kappa **{report['cohen_kappa']:.2f}**.",
        "",
        "| human \\ judge | " + " | ".join(LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
    ]
    for h in LABELS:
        lines.append(f"| {h} | " + " | ".join(str(report["confusion"][h][j]) for j in LABELS) + " |")
    if report["disagreements"]:
        lines += ["", "Disagreements:", ""]
        for d in report["disagreements"]:
            lines.append(f"- {d['id']} (prompt {d['prompt']}): human {d['human']}, judge {d['judge']} – {d['judge_reasoning']}"
                         + (f" Human note: {d['note']}" if d["note"] else ""))
    (RESULTS / "judge_calibration.md").write_text("\n".join(lines) + "\n")
    return report

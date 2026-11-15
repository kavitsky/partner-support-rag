"""Blind human grading of generated answers, used to calibrate the LLM judge.

The judge's verdict is never shown. Grades are saved after every answer, so you can stop and resume.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from evals.common import RESULTS, ROOT, load_testset

GRADES = ROOT / "evals" / "human_grades.jsonl"
KEYS = {"c": "correct", "p": "partially_correct", "i": "incorrect"}


def answer_sha(answer: str) -> str:
    return hashlib.sha256(answer.encode()).hexdigest()[:12]


def load_grades() -> list[dict]:
    return [json.loads(line) for line in GRADES.open(encoding="utf-8")] if GRADES.exists() else []


def queue(versions: list[str]) -> list[dict]:
    """Answers from the first version, then answers from later versions that differ from it."""
    seen_sha, out = set(), []
    for v in versions:
        path = RESULTS / f"answers_{v}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            sha = answer_sha(r["answer"])
            if (r["id"], sha) in seen_sha:
                continue
            seen_sha.add((r["id"], sha))
            out.append({**r, "answer_sha": sha})
    return out


def run(versions: list[str], grader: str) -> None:
    items = {it["id"]: it for it in load_testset()}
    done = {(g["id"], g["answer_sha"]) for g in load_grades()}
    todo = [r for r in queue(versions) if (r["id"], r["answer_sha"]) not in done]
    print(f"{len(done)} answers graded so far, {len(todo)} to go. Keys: c = correct, p = partially correct, i = incorrect, s = skip, q = quit.\n")
    print("Grade against the reference answer: correct = all essential facts and nothing wrong;")
    print("partially correct = some essential facts missing or a minor slip; incorrect = wrong, wrong API, or refused when an answer exists.\n")
    for n, r in enumerate(todo, 1):
        it = items[r["id"]]
        print("=" * 100)
        print(f"[{n}/{len(todo)}] {r['id']} ({it['type']}, prompt {r['prompt_version']})")
        print(f"\nQUESTION: {it['question']}")
        print(f"\nREFERENCE: {it['expected_answer']}")
        print(f"\nANSWER:\n{r['answer']}\n")
        while True:
            key = input("grade [c/p/i/s/q]: ").strip().lower()
            if key in KEYS or key in ("s", "q"):
                break
        if key == "q":
            break
        if key == "s":
            continue
        note = input("note (optional, Enter to skip): ").strip()
        with GRADES.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "prompt_version": r["prompt_version"],
                        "answer_sha": r["answer_sha"],
                        "grade": KEYS[key],
                        "note": note,
                        "grader": grader,
                        "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\nSaved to {GRADES.relative_to(ROOT)}. Next: make calibrate")

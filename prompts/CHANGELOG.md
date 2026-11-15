# Prompt changelog

Prompts are files, versioned by folder. Every answer and judge record stores the prompt version and a short SHA of its text, so results can always be traced to the exact prompt. The active versions are set in `config.toml`.

## answer/v2 (active)

Changes from v1, each motivated by a v1 eval failure:

- **"No", a limitation, or "deprecated, use X" counts as an answer.** v1 answered "not in the docs" for q17 ("Can the MCP server create reservations?") while quoting "all tools are read-only", and for q08 (subscribing to the legacy `reservation.new`).
- **Name the replacement and retirement date** when the question uses a legacy event, endpoint or feature. Targets the outdated-version traps (q08, q10).
- **Source headers carry the API name and update date**, and the prompt says not to mix the Open API and Booking Engine API. Targets the similar-documents traps (q03, q05, q12, q13).
- **Copy identifiers and numbers exactly**, so key-fact checks and developers see `reservation.created.v2`, not a paraphrase.
- **Answer partially when only part is covered**, and put the gap in `missing`, instead of all-or-nothing.
- **Answer in the language of the question.** Gemini 2.5 Flash Lite still answers the Ukrainian q27 in English, so this rule is not reliably followed.

Effect on 28 questions (same retrieval, same judge): judge correctness 0.911 → 0.929, abstention accuracy 0.929 → 0.964, source hit 0.917 → 0.958, key-fact recall 0.778 → 0.819, faithfulness 0.995 → 1.000. Fixed q08, q17 and q18. q10 moved from an off-topic answer (partially correct) to "not in the docs" (incorrect), because retrieval never returns the migration page; see REPORT.md.

## answer/v1

Baseline: use only the sources, split into claims, cite source IDs per claim, return `not_in_docs` when the answer is missing.

## judge/correctness/v1 (active)

Three labels (correct, partially_correct, incorrect) against a human-written reference answer, reasoning before the label. Abstaining is correct only for questions whose reference says "not in the documents". Its agreement with human grades is measured in `results/judge_calibration.md`.

## judge/faithfulness/v1 (active)

Per-claim supported / not_supported verdicts, judged only against the sources each claim cites.

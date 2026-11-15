# Test set

`testset.jsonl`: 28 questions about Guesty's public Open API and Booking Engine API docs.

| Type | n | What it tests |
|---|---|---|
| answerable | 16 | Facts stated in one or two pages |
| trap_similar | 4 | Both doc sets cover the topic with different answers: rate limits (15/120/5,000 vs 5/275/16,500), token endpoint and scope, and GuestyPay token reuse (supported in the Open API flow, not supported in the Booking Engine flow; the two pages are flagged as near-duplicates at Jaccard 0.55) |
| trap_outdated | 2 | The question names a legacy resource (`reservation.new` webhook, Booking Engine `POST /api/reservations`); the answer must say it is deprecated and name the replacement and date |
| unanswerable | 4 | Not in the docs: GraphQL, an uptime SLA percentage (the rate-limits page mentions "maximum uptime", a lexical lure), pricing, QuickBooks sync. Checked with full-text search over the corpus |
| multilingual | 2 | Ukrainian questions over English docs |

## Fields

- `question`, `type`, `surface` (optional `open_api` / `booking_engine`, used as the doc-set filter, like the triage card's `surface` in Homework 1).
- `expected_answer`: the reference answer for the judge, written from the doc text.
- `key_facts`: strings that must appear in the answer's claims; `a|b` means either. Checked deterministically.
- `must_not_contain`: strings that would show the wrong doc set was used (e.g. Open API numbers in a Booking Engine answer).
- `expected_docs`: an answer must cite at least one (source hit).
- `evidence`: `(doc_id, quote)` pairs. A retrieved chunk is relevant if it is from that doc and contains the quote. This keeps labels valid across chunking strategies. Every quote is checked to occur in at least one chunk under both strategies.
- `reviewed_by_human`: set to `true` once a person has checked the row against the docs.

## How it was built and reviewed

Questions, references and quotes were drafted by the assistant from the extracted document text. Every quote was verified programmatically against the chunks, and the absence of each unanswerable topic was checked with full-text search. Labels corrected after inspecting retrieval misses:

- q06: the label only accepted `"expires_in": 86400`, but two other chunks state "expire after 24 hours". Both were added as evidence.
- q10: the replacement table chunk was added as evidence alongside the deprecation schedule.
- q17, q28: "cannot create, update, or delete" was added as an accepted wording of "read-only".

Human review status: see `reviewed_by_human` on each row.

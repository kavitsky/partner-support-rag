# Report

The setup: 50 Guesty developer-doc pages (36 HTML, 8 PDF, 6 DOCX) split into 579 structure-aware chunks, hybrid BM25 and vector search with cross-encoder reranking, and answers from Gemini 2.5 Flash Lite (prompt v2) with per-claim citations. Evaluated on 28 questions.

**Summary.** Answers are faithful to their sources (1.00), say "not in the docs" exactly when they should on 27 of 28 questions, and are judged correct on 0.93 on a 0–1 scale. The remaining errors come from retrieval: tables and migration pages rank too low, and Ukrainian questions miss English chunks. They don't come from made-up content. Answering costs about $0.21 per 1,000 questions.

## 1. Ingestion

| | |
|---|---|
| Documents | 50 (HTML 36, PDF 8, DOCX 6); 38 Open API, 12 Booking Engine API |
| Exact duplicates (normalized SHA-256) | 0 |
| Near-duplicates (5-shingle Jaccard ≥ 0.5) | 1 pair, kept: GuestyPay tokenization in Open API and Booking Engine (0.55), which contradict each other on token reuse |
| Chunks (structure / fixed) | 579 / 534 |

| Strategy | Mean tokens | p95 | Starts at a block boundary | Ends at a block boundary | Cuts a table or code block |
|---|---|---|---|---|---|
| structure | 222 | 350 | 96.7% | 79.3% | 1.4% |
| fixed 300/50 | 290 | 300 | 53.4% | 23.8% | 21.2% |

PDF chunks end at a block boundary less often (63% versus 80% for HTML), because paragraph and heading reconstruction from printed pages is imperfect.

## 2. Retrieval

24 answerable questions. A chunk counts as relevant if it comes from the evidence document and contains the evidence quote.

| Configuration | recall@5 | MRR@10 | doc recall@5 | ms/query |
|---|---|---|---|---|
| structure / BM25 | 0.792 | 0.694 | 0.875 | 0.7 |
| structure / vector (bge-small) | 0.750 | 0.619 | 0.812 | 7 |
| structure / hybrid (RRF) | 0.792 | 0.720 | 0.833 | 5 |
| structure / hybrid + rerank | 0.875 | 0.809 | 0.938 | 958 |
| **structure / hybrid + rerank + doc-set filter (production)** | **0.875** | **0.809** | **0.938** | 1,081 |
| fixed / BM25 | 0.833 | 0.615 | 0.917 | 0.8 |
| fixed / vector | 0.667 | 0.537 | 0.750 | 8 |
| fixed / hybrid | 0.833 | 0.642 | 0.812 | 6 |
| fixed / hybrid + rerank | 0.833 | 0.670 | 0.875 | 992 |
| structure / vector (multilingual MiniLM) | 0.625 | 0.472 | 0.708 | 27 |
| structure / hybrid + rerank (multilingual MiniLM) | 0.917 | 0.810 | 0.958 | 1,049 |

What the comparison shows:

- **Reranking matters most:** with structure-aware chunks, MRR rises from 0.72 to 0.81.
- **Hybrid search isn't better than BM25 alone on recall here:** the questions are full of exact identifiers (`reservation.created.v2`, `/reservations-v3`), which BM25 handles well. Hybrid does improve MRR, and it gives the reranker a better candidate pool.
- **Structure-aware chunking ranks the relevant chunk higher in every mode,** and its recall is best once the reranker is on. Fixed windows get slightly higher recall@5 without the reranker, thanks to their overlap.
- **The doc-set filter didn't change retrieval metrics,** because the reranker already demotes the other API's pages for these questions. It's kept because it guarantees the answer can't mix in the other API's facts.
- **The multilingual embedding model is worse on its own** (0.63 vs 0.75) but best combined with hybrid search and reranking (0.917). The gain is one question (q08). With only 24 questions that isn't enough to switch models.

## 3. Answers

28 questions, judged by GPT-4.1-mini. Prompt v1 vs v2, with the same retrieval:

| Metric | v1 | v2 | Threshold |
|---|---|---|---|
| Abstention accuracy | 0.929 | **0.964** | 0.85 |
| Source hit (cites an expected doc) | 0.917 | **0.958** | 0.80 |
| Key-fact recall | 0.778 | **0.819** | 0.70 |
| Citation validity | 1.000 | 1.000 | 0.95 |
| Judge correctness (1 / 0.5 / 0) | 0.911 | **0.929** | 0.70 |
| Faithfulness (claims supported by cited sources) | 0.995 | **1.000** | 0.85 |
| Contains the other API's facts | 0.000 | 0.000 | – |

v2 by question type:

| Type | n | Abstention | Source hit | Key facts | Judge | Faithfulness |
|---|---|---|---|---|---|---|
| answerable | 16 | 1.000 | 1.000 | 0.854 | 1.000 | 1.000 |
| trap_similar | 4 | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 |
| trap_outdated | 2 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 |
| unanswerable | 4 | 1.000 | – | – | 1.000 | – |
| multilingual | 2 | 1.000 | 1.000 | 0.500 | 0.750 | 1.000 |

The abstain gate (rerank score below -4.0) didn't fire on any test question. All 4 unanswerable questions were refused by the model itself. No answer needed a repair attempt or failed validation.

## 4. Judge calibration

**Pending human grading.** Run `make grade` (blind: the judge's verdict is hidden), grade at least 12 answers, then run `make calibrate`. That writes `results/judge_calibration.md`, whose numbers go here:

| | |
|---|---|
| Answers graded by a human | – |
| Exact agreement (3 labels) | – |
| Binary agreement (correct vs not) | – |
| Cohen's kappa | – |

One judge error is already known. For v1 q17 the assistant wrongly said "not in the docs", and the judge rated it *correct* even though its prompt says that abstaining when the reference has an answer is incorrect (see error 5).

## 5. Typical errors

1. **The deprecated-endpoint question gets "not in the docs" (q10, retrieval).** The Booking Engine migration page explains that `POST /api/reservations` is gone and lists its replacements in a table. The cross-encoder scores that table chunk at 0.86, below how-to prose about "creating reservations", so it never reaches the top 5. Given the context it received, v2 correctly said "not in the docs". v1 answered with unrelated activation steps instead. Cause: rerankers trained on prose under-rate tables, and the question uses the old endpoint's vocabulary.
2. **The Ukrainian rate-limit question misses the numbers (q27, retrieval and extraction).** "Які ліміти запитів має Open API?" retrieves the rate-limits intro but not the chunk with the numbers. In the PDF that table is flattened into one line ("Maximum Number of Requests Time Frame 15 one second 120 one minute 5,000 one hour"), which matches neither the English nor the Ukrainian wording well. The answer is accurate but has no limits, and it is in English despite v2's language rule.
3. **A claim cites the similar page from another provider (q13, citation).** The Open API GuestyPay answer cites both the GuestyPay page and the *Stripe* tokenization page for "set reuse to true". It's correct here because the two pages agree, but the same behaviour would give a wrong answer where they differ: the Booking Engine GuestyPay page says reuse is *not* supported. Faithfulness can't catch this, because the claim *is* supported by the Stripe page.
4. **Section labels in citations can be imprecise (q08, metadata).** Small sections are merged into their neighbour and the chunk keeps the first section's name, so a correct claim about `reservation.new` is cited as "Reservation webhooks › Guest folio updates". The document is right but the section isn't.
5. **The judge is lenient about wrong refusals (v1 q17, judge).** The answer said "not in the docs", then quoted "all tools are read-only", and the judge called it correct. A judge that rewards hedged refusals would hide over-refusal regressions. That's why human calibration is required before trusting the judge's score.

## 6. Next improvements

- **Tables for the reranker:** render table rows as sentences ("/api/reservations is replaced by /api/reservations/quotes …") in the text the reranker scores, or try a stronger multilingual reranker (bge-reranker-v2-m3). This targets errors 1 and 2.
- **Cross-lingual questions:** translate non-English questions to English before retrieval, or adopt the multilingual embedder after a larger test, and enforce the answer language with a check.
- **Track every section in merged chunks** and cite the section that contains the quoted sentence (error 4).
- **Judge prompt v2:** add few-shot examples of hedged refusals, and recalibrate against the human grades (error 5).
- **Version awareness:** use `updated_at` and legacy markers as a ranking signal when the question names a legacy resource, and add more outdated-version traps (only 2 today).
- **A bigger, real test set:** turn Homework 1's partner tickets into questions and set per-type thresholds, so traps can't hide behind the overall average.

## 7. Cost

| | Per question | Per 1,000 questions |
|---|---|---|
| Answer (Gemini 2.5 Flash Lite, 1,494 in / 157 out tokens) | $0.00021 | **$0.21** |
| Evaluation judge (GPT-4.1-mini, 2 calls) | $0.00065 | $0.65 |
| Embeddings and reranking (local ONNX) | $0 | $0 |

Mean context: 1,054 tokens of sources out of the 2,000-token budget. Reranking on CPU dominates latency (about 1 s per question on a laptop).

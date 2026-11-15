# partner-support-rag

A question-answering assistant over Guesty's public developer docs (Open API and Booking Engine API). It answers only from retrieved passages, cites a document and section for every claim, and says "not in the docs" instead of guessing. It comes with retrieval and answer evals that run in CI.

Built for the AI Engineering course, block 2 ("RAG on your own data + Evals"). It continues [partner-ticket-extractor](https://github.com/kavitsky/partner-ticket-extractor): that project turns a partner's support ticket into a triage card, and this one answers the API question in it. The card's `surface` field maps to this project's `doc_set` filter.

Results, typical errors and next steps are in [REPORT.md](REPORT.md).

## Quick start

```bash
cp .env.example .env          # add OPENROUTER_API_KEY
make all                      # install, fetch docs, ingest, retrieval comparison, answer evals
```

Requires Python 3.12 and Google Chrome or Chromium (used to print the PDF documents; without it those documents fall back to HTML). The first run downloads about 50 pages and about 400 MB of ONNX models. Embeddings and LLM responses are cached in `.cache/`, so reruns are fast and free.

```bash
.venv/bin/python main.py ask "What are the Booking Engine API rate limits?" --doc-set booking_engine
.venv/bin/python main.py search "reservation.created.v2" --mode bm25 --updated-after 2026-01-01 --format pdf html
make check                    # what CI runs: retrieval + answer evals, exit 1 below thresholds
make grade && make calibrate  # hand-grade answers blind, then measure judge agreement
make test                     # offline unit tests
```

## Corpus and storage policy

50 pages listed in [`data/sources.toml`](data/sources.toml): 38 from the [Open API guides](https://open-api-docs.guesty.com/docs) and 12 from the [Booking Engine API guides](https://booking-api-docs.guesty.com/docs). They cover authentication, rate limits, webhooks, reservations, payments, migrations and the MCP server.

The docs are public, but this repo doesn't redistribute their text. `make fetch` downloads them at run time into `data/raw/` (gitignored), and CI caches the download. What is committed is [`data/manifest.json`](data/manifest.json): URL, title, `updatedAt` from the page, format, raw-file SHA-256, normalized content hash, token count, duplicate flags and chunk counts. The answer logs in `results/` keep chunk IDs and titles but not source text.

Three formats, each with its own extractor:

| Format | n | How it is produced | Extraction |
|---|---|---|---|
| HTML | 36 | The live page | The ReadMe article body only, with navigation, header, buttons and "Did this page help you?" removed. Headings, lists, tables and code are kept as Markdown |
| PDF | 8 | The live page printed with headless Chrome | Repeated headers and footers, browser date and URL lines, and navigation titles are removed; JSON lines are grouped back into code; heading lines and paragraphs are rebuilt |
| DOCX | 6 | The page's Markdown export converted to Word (python-docx): headings, lists, tables and code styles | Walks paragraphs and tables in order and maps Heading/List styles back to Markdown |

The PDFs are realistic, so they are messy. Tables come out as one line, some headings land out of order, and collapsed `<details>` content isn't printed. Because of that last point the FAQ page is HTML rather than PDF.

## Ingestion

[`rag/ingest.py`](rag/ingest.py) extracts each document, cleans and normalizes it (Unicode NFKC, zero-width characters, non-breaking spaces, whitespace), then deduplicates:

- **Exact duplicates:** SHA-256 of the lowercased, whitespace-collapsed text; duplicates are dropped. The current corpus has none.
- **Near-duplicates:** Jaccard similarity of word 5-shingles, at or above 0.5. They are flagged in the manifest but kept. The one pair found is the GuestyPay tokenization page in both doc sets (0.55), and the two versions contradict each other on token reuse. Dropping one would delete a correct answer, so both are kept and the doc-set filter and prompt tell them apart. The pair is a test-set trap (q12/q13).
- **Change detection:** `content_hash` is pinned in the manifest, and ingest reports documents whose text changed since then.

It writes `data/processed/chunks_structure.jsonl`, `chunks_fixed.jsonl` and `documents.jsonl`, plus [`results/ingest_report.json`](results/ingest_report.json).

### Chunking: structure-aware, compared with fixed-size

Docs pages are made of short sections under headings, with tables (rate limits, endpoint mappings) and code samples (token requests, payloads). A fixed window ignores all of that. Structure-aware chunking:

1. Cuts at headings (H1–H4) and tracks the heading path (`Rate Limits > Example After a Single Request`).
2. Inside a section, packs whole blocks (paragraph, list, table, code fence) up to 350 tokens (cl100k_base).
3. Splits an oversized block by its own structure. Tables split by rows with the header row repeated, code by lines inside its fence, prose by sentences. Fixed token windows are the last resort.
4. Merges a section under 60 tokens into its neighbour, so a one-line section doesn't become a chunk without context.

Each chunk is indexed as `heading path + text`, so a chunk that only says "Step by step:" still carries its page and section. The baseline is a fixed 300-token window with 50 tokens of overlap.

| Strategy | Chunks | Mean tokens | Starts at a block boundary | Ends at a block boundary | Cuts a table or code block |
|---|---|---|---|---|---|
| structure | 579 | 222 | 96.7% | 79.3% | 1.4% |
| fixed 300/50 | 534 | 290 | 53.4% | 23.8% | 21.2% |

Retrieval quality per strategy (24 answerable questions, see below):

| Search | Structure-aware recall@5 / MRR@10 | Fixed-size recall@5 / MRR@10 |
|---|---|---|
| BM25 | 0.792 / 0.694 | 0.833 / 0.615 |
| vector | 0.750 / 0.619 | 0.667 / 0.537 |
| hybrid | 0.792 / 0.720 | 0.833 / 0.642 |
| hybrid + rerank | **0.875 / 0.809** | 0.833 / 0.670 |

With BM25 and plain hybrid search, fixed windows find a relevant chunk in the top 5 slightly more often: overlapping windows give a second chance to match. Structure-aware chunks rank the relevant chunk higher (MRR is better in every mode) and win clearly once the reranker is used, which is the production setup. The answer also gets whole tables and code samples instead of fragments.

### Chunk metadata

Each line of `chunks_*.jsonl` has: `chunk_id`, `doc_id`, `title`, `section`, `heading_path`, `url` with a section anchor, `doc_set`, `format`, `updated_at` (the page's `updatedAt`), `doc_hash`, `chunk_hash`, `tokens`, `strategy` and `text`.

## Search

[`rag/search.py`](rag/search.py):

- **Vector:** fastembed `BAAI/bge-small-en-v1.5` (ONNX, CPU) stored in an in-memory Qdrant collection with cosine distance. Embeddings are cached on disk by `sha256(model + text)`, so re-ingesting re-embeds only changed chunks.
- **BM25:** `rank_bm25` with a tokenizer that keeps API identifiers whole and also indexes their parts. `reservation.created.v2` yields `reservation.created.v2`, `reservation`, `created`, `v2`.
- **Hybrid:** reciprocal rank fusion (k=60) of the top 20 results from each.
- **Rerank:** cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` over the 20 fused candidates, keeping the top 5.
- **Metadata filters:** `doc_set`, `updated_after` and `format`. They are applied inside Qdrant (payload filters; dates stored as ordinal days) and as the same predicate on BM25 results.

All 11 configurations are in [`results/retrieval.md`](results/retrieval.md) and REPORT.md. They compare BM25, vector, hybrid and hybrid with rerank, both chunking strategies, two embedding models, and search with and without the filter.

## Answers

[`rag/answer.py`](rag/answer.py) with `google/gemini-2.5-flash-lite` through OpenRouter:

1. Retrieve 5 chunks, using the `doc_set` filter when the API is known.
2. **Abstain gate:** if the best rerank score is below -4.0, answer "not in the docs" without calling the model. The cutoff was set from the score distribution and only catches clearly off-topic questions. Rerank scores can't separate answerable from unanswerable questions (the unanswerable QuickBooks question scores above several answerable ones), so the main decision is the model's, guided by the prompt.
3. **Token budget:** sources are packed best-first into 2,000 tokens. Each has a header with title, section, API and update date. The best chunk is truncated rather than dropped.
4. **Structured output:** `{status: answered | not_in_docs, claims: [{text, sources: [S1, ...]}], missing}`, enforced with a JSON schema. The code then checks that every claim cites at least one source and only sources that were provided. Otherwise the model gets one repair attempt.
5. Rendered with a citation per claim: `- claim [S1: Rate Limits › Example After a Single Request]`.

Prompts live in `prompts/answer/v1`, `v2` and `prompts/judge/...`, with a [changelog](prompts/CHANGELOG.md).

## Evals

- **Test set:** 28 questions in [`evals/testset.jsonl`](evals/testset.jsonl): answerable, unanswerable, similar-document traps, outdated-version traps and Ukrainian questions. See [`evals/TESTSET.md`](evals/TESTSET.md).
- **Retrieval** ([`evals/retrieval_eval.py`](evals/retrieval_eval.py)): recall@5 (a relevant chunk in the top 5), MRR@10 and document recall@5. A chunk is relevant if it comes from the evidence document and contains the evidence quote, so labels don't depend on chunk IDs.
- **Deterministic answer checks** ([`evals/answer_eval.py`](evals/answer_eval.py)):
  - abstention accuracy (abstained exactly when the question is unanswerable)
  - source hit (cites an expected document)
  - key-fact recall
  - citation validity
  - forbidden strings from the other API
- **LLM judge:** `openai/gpt-4.1-mini`, a different model family from the answer model. It grades correctness against the reference answer (correct, partially correct, incorrect) and faithfulness of each claim against only the sources it cites.
- **Judge calibration:** `make grade` shows answers without the judge's verdict and records human grades in `evals/human_grades.jsonl`. `make calibrate` reports exact agreement, binary agreement and Cohen's kappa in `results/judge_calibration.md`.
- **Thresholds** in `config.toml`: `make check` exits with code 1 below them.

## CI

[`.github/workflows/evals.yml`](.github/workflows/evals.yml) runs on every push and pull request. Its steps: fetch docs (PDFs with the runner's Chrome), ingest, unit tests, a retrieval check (recall@5 ≥ 0.75, MRR@10 ≥ 0.65) and the answer evals against the thresholds. Docs, models, embeddings and LLM responses are cached, so a push that doesn't change prompts, questions or retrieved context makes no paid calls. Result tables go to the job summary and an artifact. It needs the `OPENROUTER_API_KEY` repository secret.

## Cost

Measured on the 28-question run:

- **Answering:** 1,494 input and 157 output tokens per question on average, or about **$0.21 per 1,000 questions** with Gemini 2.5 Flash Lite ($0.10 / $0.40 per million tokens). Embedding and reranking run locally and cost nothing.
- **Judging:** two GPT-4.1-mini calls per question, about $0.65 per 1,000 questions, paid only when evals run on new answers.

## Layout

```
config.toml            all experiment settings and thresholds
data/sources.toml      the corpus (URLs and formats)
data/manifest.json     per-document metadata and hashes (no text)
rag/                   fetch, ingest, search, llm, answer
prompts/               answer and judge prompts, CHANGELOG.md
evals/                 test set, retrieval and answer evals, grading, calibration
results/               metric tables and answer logs (no source text)
tests/                 offline unit tests
```

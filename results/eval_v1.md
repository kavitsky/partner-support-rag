Answer model `google/gemini-2.5-flash-lite`, prompt v1 (bf50ddf0115c); judge `openai/gpt-4.1-mini`; retrieval structure / hybrid_rerank / BAAI/bge-small-en-v1.5.

| Slice | n | abstention_accuracy | source_hit | key_fact_recall | citation_validity | judge_correctness | faithfulness | forbidden_rate |
|---|---|---|---|---|---|---|---|---|
| **overall** | 28 | 0.929 | 0.917 | 0.778 | 1.000 | 0.911 | 0.995 | 0.000 |
| answerable | 16 | 0.938 | 0.938 | 0.854 | 1.000 | 0.969 | 0.993 | 0.000 |
| multilingual | 2 | 1.000 | 1.000 | 0.500 | 1.000 | 0.750 | 1.000 | 0.000 |
| trap_outdated | 2 | 0.500 | 0.500 | 0.000 | 1.000 | 0.500 | 1.000 | 0.000 |
| trap_similar | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 | 0.000 |
| unanswerable | 4 | 1.000 | – | – | – | 1.000 | – | 0.000 |

Cost: answers $0.0059, judge $0.0204; **$0.21 per 1,000 questions** (answering only); mean 1310 input / 199 output tokens.

Thresholds: abstention_accuracy ≥ 0.85, source_hit ≥ 0.8, key_fact_recall ≥ 0.7, citation_validity ≥ 0.95, judge_correctness ≥ 0.7, faithfulness ≥ 0.85.
Result: **PASS**

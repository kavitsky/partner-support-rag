Retrieval on 24 answerable questions.

| Configuration | recall@5 | MRR@10 | doc recall@5 | ms/query |
|---|---|---|---|---|
| structure / bm25 | 0.792 | 0.694 | 0.875 | 0.7 |
| structure / vector | 0.750 | 0.619 | 0.812 | 7.2 |
| structure / hybrid | 0.792 | 0.720 | 0.833 | 5.2 |
| structure / hybrid_rerank | 0.875 | 0.809 | 0.938 | 957.6 |
| fixed / bm25 | 0.833 | 0.615 | 0.917 | 0.8 |
| fixed / vector | 0.667 | 0.537 | 0.750 | 7.7 |
| fixed / hybrid | 0.833 | 0.642 | 0.812 | 5.7 |
| fixed / hybrid_rerank | 0.833 | 0.670 | 0.875 | 992.3 |
| structure / hybrid_rerank + surface filter | 0.875 | 0.809 | 0.938 | 1080.6 |
| structure / vector (paraphrase-multilingual-MiniLM-L12-v2) | 0.625 | 0.472 | 0.708 | 26.9 |
| structure / hybrid_rerank (paraphrase-multilingual-MiniLM-L12-v2) | 0.917 | 0.810 | 0.958 | 1048.5 |

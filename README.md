# PaperLens — Transformer Retrieval + Reranking

CPU-first research-paper search with BM25, pretrained sentence embeddings, weighted reciprocal rank fusion (RRF), cross-encoder reranking, and a **development-tuned / held-out SciFact benchmark workflow**.

## Run the search app

Python 3.10+; run from the repository root.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
# Optional: install CPU-only PyTorch first to avoid unnecessary CUDA packages.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The app defaults to **transformer hybrid retrieval + neural reranking**. No paid API key or GPU is needed. First use downloads encoder/reranker weights. Document embeddings persist in `.cache/paperlens` and are invalidated when paper content/order or model settings change. The encoder still loads on restart to encode new queries. Repeated queries have a bounded in-memory embedding cache.

For a lightweight offline baseline, install `requirements-lite.txt`, select LSA or BM25 and disable neural reranking. The separate browser demo in `dist/` remains an offline BM25/LSA demo; it does **not** execute the neural pipeline. The Streamlit application is the full neural application.

## Retrieval pipeline

1. **BM25** (k1=1.5, b=0.75), using sparse inverted postings.
2. **SentenceTransformer** `sentence-transformers/all-MiniLM-L6-v2`: normalized document/query embeddings, cosine similarity.
3. **Hybrid fusion**: `(1-alpha)/(60+lexical_rank) + alpha/(60+dense_rank)`, using a configurable top-N pool from each retriever.
4. **CrossEncoder** `cross-encoder/ms-marco-MiniLM-L-6-v2`: reranks the top fused candidates; `score` then means the reranker score, while `retrieval_score` retains the fusion score.
5. Return the top results with paper metadata and score inspection.

CPU is the default. Candidate pools of 20 or 50 are sensible starting points; reranking more candidates adds inference cost. Dense retrieval is exhaustive, not FAISS: vectors and corpus live in RAM. Model token limits can truncate long abstracts. This project searches titles and abstracts/summaries, not full PDFs. The cross-encoder has its own truncation limit.

## Held-out benchmark: BEIR SciFact

```bash
python scripts/prepare_scifact.py
python -m paperlens.benchmark tune --candidate-counts 20 50
python -m paperlens.benchmark test
```

The preparation script downloads the official BEIR archive and verifies its published checksum. It preserves the **entire corpus**, selects up to 100 development queries in a deterministic hash order from official **training qrels**, and retains **all official test queries**. Development queries with matching test IDs or exact normalized text are excluded. Manifest fingerprints protect the prepared inputs.

`tune` sees only development query texts and relevance judgments. It compares fusion weights 0.25 / 0.50 / 0.75 and candidate counts 20 / 50, selecting by development nDCG@10. The selected configuration is saved in `reports/scifact/config.json`.

`test` loads that frozen configuration; CLI tuning/model flags do not override it. It verifies corpus/split fingerprints and disjoint query IDs/texts, then compares:

- BM25
- Transformer dense retrieval
- Hybrid retrieval
- Hybrid + cross-encoder reranking

For matching the measured environment, install `torch==2.5.1` from the CPU wheel index and then `requirements-benchmark.txt`.

Outputs include Recall@10, graded nDCG@10, MRR@10, median/p95 query latency, ranked document IDs, per-query metrics, model identifiers/revisions, package versions and indexing/loading time. Query embedding caches are cleared between test methods; warmup/model loading is excluded from search timings. Model revision flags accept pinned Hugging Face revisions for reproducibility:

```bash
python -m paperlens.benchmark tune --model-revision YOUR_ENCODER_COMMIT --reranker-revision YOUR_RERANKER_COMMIT
```

A held-out test means it was excluded from **this project's tuning**; it does not prove absence from pretrained model data. Do not rerun tuning decisions based on test scores. Do not claim statistical significance from a small absolute metric difference. Use per-query outputs for failure analysis and paired uncertainty estimation.

If SciFact is already extracted, skip downloading:

```bash
python scripts/prepare_scifact.py --source /path/to/scifact
```

For CPU verification without model downloads, `tune --backend lsa --no-rerank` followed by `test` checks the split/config workflow but **is not** the transformer/reranker experiment.

## Small development smoke test

The bundled `data/papers.json` has 22 real paper titles/arXiv links and author-written short summaries. Its eight labeled queries are a toy development set. The original `reports/evaluation.json` contains the previous LSA run; these figures are **not** held-out SciFact results.

```bash
python -m paperlens.evaluate --rerank
python -m unittest discover -s tests
```

The evaluation command compares base methods separately and adds reranking as a fourth method when requested. It labels outputs as development results.

## Add your own corpus

```bash
python scripts/fetch_arxiv.py --query "cat:cs.CL" --limit 500 --output data/arxiv.json
```

Upload that JSON in Streamlit. The importer retrieves metadata/abstracts, not PDFs. Another BEIR dataset can be converted using `scripts/convert_beir.py`; the held-out runner requires `papers.json`, `dev_queries.json`, `test_queries.json` and a matching manifest like the SciFact preparer produces.

## Files

| File | Purpose |
|---|---|
| `paperlens/engine.py` | Sparse/dense retrieval, persistent embedding cache, rank fusion, reranking |
| `paperlens/benchmark.py` | Development tuning and frozen held-out evaluation |
| `scripts/prepare_scifact.py` | Official archive download, full corpus, leakage checks and split fingerprints |
| `paperlens/evaluate.py` | Metrics and ad hoc development evaluation |
| `paperlens/app.py` | Full Streamlit search application |
| `tests/` | Ranking, model/cache contracts, filters, metrics and held-out workflow |
| `dist/` | Lightweight offline browser demo |

## Resume wording

“Built a CPU-compatible paper-search pipeline combining BM25 and pretrained transformer embeddings with reciprocal rank fusion and cross-encoder reranking; evaluated with development-only tuning and held-out SciFact queries.”

Add achieved metrics and improvements only from an executed held-out report; no improvement is guaranteed merely by adding a reranker.

## References and licenses

- BEIR: https://github.com/beir-cellar/beir
- Retrieve & Re-rank: https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- arXiv API: https://info.arxiv.org/help/api/user-manual.html

Code is MIT licensed. External datasets, paper metadata and model weights retain their respective terms.

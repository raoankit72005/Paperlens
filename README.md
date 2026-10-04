# PaperLens — Hybrid Research Paper Retrieval

A reproducible CPU-first retrieval project with **BM25**, **dense retrieval**, **weighted reciprocal rank fusion**, an optional **cross-encoder reranker**, a Streamlit application, and a standalone browser demo.

## Quick start

Python 3.10+ recommended. Run commands from the project root.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

No API key or GPU is required for the default LSA baseline. To use neural retrieval and reranking:

```bash
python -m pip install -r requirements-neural.txt
```

Select `transformer` in the app. First use downloads model weights and requires internet. The sentence encoder is `sentence-transformers/all-MiniLM-L6-v2`; the optional reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. CPU inference is supported but may be slow for large corpora. Model downloads and the neural paths were not executed in the build environment; the LSA path was tested.

## What the project implements

- BM25 with k1=1.5, b=0.75 over title + summary.
- Offline LSA: TF-IDF → 16-dimensional truncated SVD → L2 normalization → cosine similarity. **LSA is not a pretrained transformer**, and is only a lightweight baseline.
- Optional pretrained sentence-transformer embeddings and cosine retrieval.
- Hybrid score: `(1-alpha)/(60+lexical_rank) + alpha/(60+dense_rank)`. Ranks start at 1; unmatched/nonpositive-score results receive no credit. Filters apply before fusion.
- Optional cross-encoder reranking over the top 50 retrieved candidates.
- Corpus upload, topic/year filtering and per-result score inspection in Streamlit.
- Recall@k, graded nDCG@k, MRR@k, median and p95 search latency.

The browser demo implements BM25 and uses the exact exported Python TF-IDF/SVD matrices for LSA queries. It searches the bundled 22-paper corpus, with no server or model download. It does **not** run the neural encoder/reranker. The full Python app does.

## Data and evaluation honesty

`data/papers.json` contains real paper titles and arXiv links, plus **author-written short summaries**, not verbatim abstracts or full PDFs. `data/queries.json` contains eight manually labeled development queries. They are useful for smoke tests only: no independent held-out evaluation, significance claim or state-of-the-art claim is supported.

The checked report at `reports/evaluation.json` records the actual run. On this small set, nDCG@10 is approximately BM25 **0.835**, LSA **0.870**, hybrid **0.840**. Hybrid does not win this run. Do not present these toy numbers as benchmark achievements on a resume.

```bash
python -m unittest discover -s tests
python -m paperlens.evaluate
python -m paperlens.evaluate --backend transformer --output reports/neural.json
python -m paperlens.evaluate --backend transformer --rerank --output reports/reranked.json
```

Latency excludes indexing and model loading. Each method currently computes both base retrieval scores; timings measure this implementation, not isolated BM25 versus encoder cost. Small timing differences should not be interpreted as meaningful. Dense retrieval is exhaustive; documents and vectors remain in RAM. This is not a million-document production index.

## Expand to real arXiv metadata

```bash
python scripts/fetch_arxiv.py --query "cat:cs.CL" --limit 500 --output data/arxiv.json
```

Upload the resulting JSON in Streamlit. This fetches metadata/abstracts, not PDFs. It paginates, retries and waits between requests. Respect arXiv access policies. This network-dependent importer was not executed here.

## Evaluate on a public benchmark

Download an appropriate BEIR dataset from its official repository, following the dataset's license and instructions. Start with a manageable set such as SciFact. Preserve all relevant documents; arbitrary sampling invalidates relevance evaluation.

```bash
python scripts/convert_beir.py /path/to/scifact --split test
python -m paperlens.evaluate --corpus data/beir/papers.json --queries data/beir/queries.json --output reports/beir-lsa.json
python -m paperlens.evaluate --corpus data/beir/papers.json --queries data/beir/queries.json --backend transformer --output reports/beir-neural.json
```

Tune fusion weights and other hyperparameters on a separate development set; keep test queries untouched. Compare BM25, LSA, sentence encoder, hybrid, hybrid + reranker. Report dataset/version, corpus size, query counts, hardware, dependency versions, indexing time and memory. Repeat latency runs after warmup. Inspect failures, including rare terms, abbreviations, ambiguous queries and missing relevant documents. The engine's `alpha` parameter enables weight ablations; use development queries for selecting it.

The BEIR converter preserves IDs and positive graded relevance labels. Benchmark documents use year=0 and no source URL; use the evaluation CLI rather than year-filtered paper UI for that converted corpus.

## Repository layout

```
paperlens/engine.py       Retrieval and reranking
paperlens/evaluate.py     Metrics and evaluation CLI
paperlens/app.py          Full Streamlit UI
streamlit_app.py          Streamlit entrypoint
data/                    Curated corpus and query judgments
scripts/fetch_arxiv.py    Metadata importer
scripts/convert_beir.py   Benchmark converter
reports/evaluation.json  Actual development results
tests/                   Ranking, filter and metric checks
dist/                    Self-contained browser demo assets
```

## Suggested resume description

“Built a CPU-compatible paper retrieval system combining BM25 and dense embeddings with reciprocal rank fusion; implemented cross-encoder reranking and reproducible retrieval evaluation.”

Add dataset sizes and measured improvements **only after completing your own held-out experiment**. Describe LSA honestly if the neural configuration has not been run.

## References

- BEIR: https://github.com/beir-cellar/beir
- Sentence Transformers: https://www.sbert.net/
- arXiv API: https://info.arxiv.org/help/api/user-manual.html
- Streamlit: https://docs.streamlit.io/

Project code is MIT licensed. Paper metadata, model weights and external datasets retain their own terms.

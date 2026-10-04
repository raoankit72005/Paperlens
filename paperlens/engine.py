"""CPU-first sparse / transformer retrieval and cross-encoder reranking."""
import collections
import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class SearchEngine:
    def __init__(self, papers, backend='transformer', model=ENCODER,
                 reranker_model=RERANKER, device='cpu', batch_size=32,
                 cache_dir='.cache/paperlens', model_revision=None,
                 reranker_revision=None):
        if not papers or len({p['id'] for p in papers}) != len(papers):
            raise ValueError('Provide a nonempty corpus with unique paper IDs')
        if batch_size < 1 or backend not in ('bm25', 'lsa', 'transformer'):
            raise ValueError('Invalid backend or batch size')
        self.papers = papers
        self.texts = [p['title'] + ' ' + p['summary'] for p in papers]
        self.backend, self.model = backend, model
        self.model_revision, self.reranker_revision = model_revision, reranker_revision
        self.reranker_model, self.device, self.batch_size = reranker_model, device, batch_size
        self.encoder = self.reranker = None
        tokens = [tokenize(t) for t in self.texts]
        self.lengths = np.array([len(t) for t in tokens])
        self.avg_length = max(float(self.lengths.mean()), 1)
        # Sparse inverted postings avoid rescanning the entire corpus for each query term.
        postings = collections.defaultdict(list)
        for i, words in enumerate(tokens):
            for word, count in collections.Counter(words).items():
                postings[word].append((i, count))
        self.postings = {w: (np.array([i for i, _ in v]), np.array([c for _, c in v]))
                         for w, v in postings.items()}
        self.idf = {w: math.log(1 + (len(papers)-len(v[0])+.5)/(len(v[0])+.5))
                    for w, v in self.postings.items()}
        self.corpus_fingerprint = hashlib.sha256(json.dumps(
            [(p['id'], t) for p, t in zip(papers, self.texts)],
            ensure_ascii=False).encode()).hexdigest()
        self.cache_hit = False
        if backend == 'transformer':
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model, device=device, revision=model_revision)
            if hasattr(self.encoder, '__getitem__'):
                self.model_revision = getattr(self.encoder[0].auto_model.config, '_commit_hash', None) or model_revision
            # Include content/order, encoder revision and tokenizer truncation settings in the key.
            key = hashlib.sha256(json.dumps([self.corpus_fingerprint, model,
                self.model_revision, self.encoder.max_seq_length, 'normalized-v1']).encode()).hexdigest()
            cache = Path(cache_dir) / (key+'.npy') if cache_dir else None
            if cache and cache.exists():
                vectors = np.load(cache, allow_pickle=False)
                if vectors.ndim == 2 and vectors.shape == (len(papers), self.encoder.get_sentence_embedding_dimension()) and np.isfinite(vectors).all():
                    self.vectors = vectors
                    self.cache_hit = True
            if not self.cache_hit:
                self.vectors = np.asarray(self.encoder.encode(self.texts,
                    batch_size=batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=True), dtype=np.float32)
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temp = cache.with_name(cache.name+f'.{os.getpid()}.tmp')
                    with temp.open('wb') as f:
                        np.save(f, self.vectors, allow_pickle=False)
                    temp.replace(cache)
        elif backend == 'lsa':
            self.vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b[a-z0-9]+\b')
            x = self.vectorizer.fit_transform(self.texts)
            dimensions = min(16, x.shape[0]-1, x.shape[1]-1)
            if dimensions < 1:
                raise ValueError('LSA requires at least two documents and two vocabulary terms')
            self.svd = TruncatedSVD(n_components=dimensions, random_state=42)
            self.vectors = normalize(self.svd.fit_transform(x))

    def bm25(self, query):
        score = np.zeros(len(self.papers))
        for word in set(tokenize(query)):
            if word not in self.postings:
                continue
            indices, tf = self.postings[word]
            score[indices] += self.idf[word]*tf*2.5/(tf+1.5*(.25+.75*self.lengths[indices]/self.avg_length))
        return score

    @lru_cache(maxsize=256)
    def query_vector(self, query):
        if self.backend == 'bm25':
            raise ValueError('Select a semantic backend for dense retrieval')
        if self.encoder is not None:
            return np.asarray(self.encoder.encode([query], batch_size=self.batch_size,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
        return normalize(self.svd.transform(self.vectorizer.transform([query])))

    def semantic(self, query):
        return (self.vectors @ self.query_vector(query).T).ravel()

    def load_reranker(self):
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.reranker_model, device=self.device,
                                         revision=self.reranker_revision)
            if hasattr(self.reranker, 'model'):
                self.reranker_revision = getattr(self.reranker.model.config, '_commit_hash', None) or self.reranker_revision
        return self.reranker

    def search(self, query, mode='hybrid', k=10, alpha=.5, topic=None,
               min_year=None, rerank=False, candidate_k=50, rrf_k=60):
        if mode not in ('bm25', 'semantic', 'hybrid') or not 0 <= alpha <= 1 or k < 1 or candidate_k < k or rrf_k < 0:
            raise ValueError('Invalid search configuration; candidate_k must be >= k')
        if not query.strip():
            return []
        eligible = [i for i, p in enumerate(self.papers)
                    if (not topic or p.get('topic') == topic)
                    and (min_year is None or p.get('year', 0) >= min_year)]
        if not eligible:
            return []
        # Only compute retrieval methods needed for the selected mode/weight.
        use_b = mode == 'bm25' or (mode == 'hybrid' and alpha < 1)
        use_s = mode == 'semantic' or (mode == 'hybrid' and alpha > 0)
        b = self.bm25(query) if use_b else None
        s = self.semantic(query) if use_s else None
        def ranked(scores):
            # Transformers rank all eligible documents, including negative cosine values;
            # lexical/LSA baselines can return an honest no-match result.
            indices = eligible if scores is s and self.backend == 'transformer' else [i for i in eligible if scores[i] > 1e-10]
            return sorted(indices, key=lambda i: (-scores[i], self.papers[i]['id']))
        fusion = np.zeros(len(self.papers))
        if mode == 'hybrid':
            for weight, values in ((1-alpha, b), (alpha, s)):
                if values is not None and weight > 0:
                    for rank, i in enumerate(ranked(values)[:candidate_k], 1):
                        fusion[i] += weight/(rrf_k+rank)
            scores = fusion
            ranking = ranked(fusion)
        else:
            scores = b if mode == 'bm25' else s
            ranking = ranked(scores)
        cross = {}
        if rerank and ranking:
            candidates = ranking[:candidate_k]
            predictions = np.asarray(self.load_reranker().predict(
                [(query, self.texts[i]) for i in candidates],
                batch_size=self.batch_size, show_progress_bar=False)).reshape(-1)
            if len(predictions) != len(candidates) or not np.isfinite(predictions).all():
                raise ValueError('Reranker returned invalid scores')
            cross = dict(zip(candidates, map(float, predictions)))
            ranking = sorted(candidates, key=lambda i: (-cross[i], self.papers[i]['id']))
        return [dict(self.papers[i], score=cross.get(i, float(scores[i])),
                     retrieval_score=float(scores[i]),
                     **({'bm25': float(b[i])} if b is not None else {}),
                     **({'semantic': float(s[i])} if s is not None else {}),
                     **({'rerank_score': cross[i]} if i in cross else {})) for i in ranking[:k]]

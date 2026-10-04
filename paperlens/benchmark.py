"""Development-only tuning followed by frozen-config, held-out test evaluation."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import time
from pathlib import Path
import numpy as np
from .engine import SearchEngine, ENCODER, RERANKER
from .evaluate import metrics


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_queries(queries, corpus):
    ids = {p['id'] for p in corpus}
    if not queries or len({q['id'] for q in queries}) != len(queries):
        raise ValueError('Query IDs must be unique and queries nonempty')
    for q in queries:
        if not q['text'].strip() or not q['relevance'] or not any(g > 0 for g in q['relevance'].values()):
            raise ValueError('Each query needs text and positive relevance judgments')
        if not set(q['relevance']).issubset(ids):
            raise ValueError('Relevant documents are missing from the corpus; do not sample the corpus')


def evaluate(engine, queries, *, mode, k, alpha, candidate_k, rerank):
    values, latencies, per_query = [], [], []
    # Warm model inference without using any query judgments.
    engine.search('scientific research', mode=mode, k=k, alpha=alpha,
                  candidate_k=candidate_k, rerank=rerank)
    for position, q in enumerate(queries, 1):
        start = time.perf_counter()
        results = engine.search(q['text'], mode=mode, k=k, alpha=alpha,
                                candidate_k=candidate_k, rerank=rerank)
        elapsed = (time.perf_counter()-start)*1000
        value = metrics(results, q['relevance'], k)
        values.append(value); latencies.append(elapsed)
        per_query.append(dict(query_id=q['id'], recall=value[0], ndcg=value[1],
                              mrr=value[2], latency_ms=elapsed, ranked_ids=[p['id'] for p in results]))
        if position % 50 == 0:
            print(f'{mode} rerank={rerank}: {position}/{len(queries)} queries evaluated', flush=True)
    avg = np.mean(values, axis=0)
    return dict(mode=mode, rerank=rerank, alpha=alpha, candidate_k=candidate_k,
        recall_at_k=float(avg[0]), ndcg_at_k=float(avg[1]), mrr_at_k=float(avg[2]),
        median_ms=float(np.median(latencies)), p95_ms=float(np.percentile(latencies,95)),
        per_query=per_query)


def provenance(engine, corpus_path):
    versions = {}
    for name in ['numpy','scikit-learn','sentence-transformers','transformers','torch']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return dict(corpus_sha256=digest(corpus_path), corpus_size=len(engine.papers),
        backend=engine.backend, encoder=engine.model, model_revision=engine.model_revision,
        reranker=engine.reranker_model, reranker_revision=engine.reranker_revision,
        device=engine.device, encoder_max_seq_length=getattr(engine.encoder,'max_seq_length',None),
        reranker_max_length=getattr(engine.reranker,'max_length',None),
        cpu_threads=__import__('torch').get_num_threads() if engine.encoder is not None else None,
        python=platform.python_version(), platform=platform.platform(),
        processor=platform.processor(), packages=versions, embedding_cache_hit=engine.cache_hit,
        notes=['Abstract/title retrieval, not full PDFs.',
               'Held-out means held out from this project tuning, not guaranteed absent from model pretraining.',
               'Search latency excludes model loading/indexing; query-vector caching may benefit later methods. Cross-encoder predictions are not cached.'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['tune','test'])
    ap.add_argument('--data',default='data/scifact')
    ap.add_argument('--config',default='reports/scifact/config.json')
    ap.add_argument('--output',default=None)
    ap.add_argument('--backend',choices=['transformer','lsa'],default='transformer')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--model',default=ENCODER)
    ap.add_argument('--model-revision',default=None)
    ap.add_argument('--reranker',default=RERANKER)
    ap.add_argument('--reranker-revision',default=None)
    ap.add_argument('--k',type=int,default=10)
    ap.add_argument('--alphas',type=float,nargs='+',default=[.25,.5,.75])
    ap.add_argument('--candidate-counts',type=int,nargs='+',default=[20,50])
    ap.add_argument('--no-rerank',action='store_true',help='Offline pipeline verification only; not the requested neural experiment')
    a=ap.parse_args();root=Path(a.data);corpus_path=root/'papers.json';papers=read(corpus_path)
    manifest=read(root/'manifest.json')
    for name in ['papers.json','dev_queries.json','test_queries.json']:
        if digest(root/name) != manifest['files'][name]: raise ValueError('Prepared data changed; rebuild dataset manifest')
    if a.phase=='tune':
        # Tuning never loads test query texts or judgments.
        queries=read(root/'dev_queries.json');validate_queries(queries,papers)
        if a.k<1 or any(not 0<=x<=1 for x in a.alphas) or any(c<a.k for c in a.candidate_counts): raise ValueError('Invalid tuning grid')
        settings=dict(backend=a.backend,model=a.model,model_revision=a.model_revision,
            reranker_model=a.reranker,reranker_revision=a.reranker_revision,
            device=a.device,batch_size=a.batch_size)
    else:
        config=read(a.config)
        if config['corpus_sha256']!=digest(corpus_path) or config['dev_sha256']!=digest(root/'dev_queries.json') or config['test_sha256']!=digest(root/'test_queries.json'): raise ValueError('Configuration belongs to different dataset/splits')
        settings=config['engine'];queries=read(root/'test_queries.json');validate_queries(queries,papers)
        dev=read(root/'dev_queries.json')
        if {q['id'] for q in dev}&{q['id'] for q in queries} or {q['text'].strip().casefold() for q in dev}&{q['text'].strip().casefold() for q in queries}: raise ValueError('Development/test query overlap')
    start=time.perf_counter();engine=SearchEngine(papers,**settings);index_seconds=time.perf_counter()-start
    report=dict(phase=a.phase,dataset=manifest['dataset'],query_count=len(queries),
        k=a.k if a.phase=='tune' else config['k'],
        provenance=provenance(engine,corpus_path),index_or_load_seconds=index_seconds,results=[])
    if a.phase=='tune':
        for alpha, count in itertools.product(a.alphas,a.candidate_counts):
            result=evaluate(engine,queries,mode='hybrid',k=a.k,alpha=alpha,candidate_k=count,rerank=not a.no_rerank)
            report['results'].append(result)
            print(f"dev alpha={alpha} candidates={count} nDCG@{a.k}={result['ndcg_at_k']:.4f}",flush=True)
        best=max(report['results'],key=lambda x:(x['ndcg_at_k'],-x['candidate_k']))
        settings.update(model_revision=engine.model_revision, reranker_revision=engine.reranker_revision)
        config=dict(engine=settings,k=a.k,alpha=best['alpha'],candidate_k=best['candidate_k'],
            rerank=not a.no_rerank,selection_metric='development nDCG@k',
            corpus_sha256=digest(corpus_path),dev_sha256=digest(root/'dev_queries.json'),test_sha256=digest(root/'test_queries.json'),
            dev_ids=[q['id'] for q in queries])
        Path(a.config).parent.mkdir(parents=True,exist_ok=True);Path(a.config).write_text(json.dumps(config,indent=2))
    else:
        report['config_sha256']=digest(a.config)
        report['k']=config['k']
        # Release query embedding cache between methods for fairer cold-query latency.
        for mode, rerank in [('bm25',False),('semantic',False),('hybrid',False)]+([('hybrid',True)] if config['rerank'] else []):
            engine.query_vector.cache_clear()
            result=evaluate(engine,queries,mode=mode,k=config['k'],alpha=config['alpha'],candidate_k=config['candidate_k'],rerank=rerank)
            report['results'].append(result)
            print(f"test {mode} rerank={rerank} nDCG@{config['k']}={result['ndcg_at_k']:.4f}",flush=True)
        report['provenance']['notes'][-1]='Latency excludes index/model loading and warmup; query cache is cleared between methods. Cross-encoder predictions are not cached.'
    notes=report['provenance']['notes']
    report['provenance']=provenance(engine,corpus_path)
    report['provenance']['notes']=notes
    output=Path(a.output or f'reports/scifact/{a.phase}.json');output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(f'Saved {output}',flush=True)

if __name__=='__main__': main()

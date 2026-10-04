"""Ad hoc development evaluation; held-out experiments use paperlens.benchmark."""
import argparse
import json
import math
from pathlib import Path


def metrics(results, relevance, k):
    if k < 1 or any(g < 0 for g in relevance.values()):
        raise ValueError('k must be positive and relevance grades nonnegative')
    ids = [p['id'] for p in results[:k]]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate retrieved IDs invalidate ranking metrics')
    relevant = {i for i, g in relevance.items() if g > 0}
    recall = len(set(ids) & relevant)/len(relevant) if relevant else 0
    dcg = sum((2**relevance.get(i, 0)-1)/math.log2(r+2) for r, i in enumerate(ids))
    ideal = sum((2**g-1)/math.log2(r+2) for r, g in enumerate(sorted(relevance.values(), reverse=True)[:k]))
    rr = next((1/(r+1) for r, i in enumerate(ids) if i in relevant), 0)
    return recall, dcg/ideal if ideal else 0, rr


def main():
    from .engine import SearchEngine
    from .benchmark import evaluate, validate_queries, provenance
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',default='data/papers.json');ap.add_argument('--queries',default='data/queries.json')
    ap.add_argument('--backend',choices=['lsa','transformer'],default='transformer')
    ap.add_argument('--k',type=int,default=10);ap.add_argument('--candidate-k',type=int,default=50)
    ap.add_argument('--alpha',type=float,default=.5);ap.add_argument('--device',default='cpu')
    ap.add_argument('--output',default='reports/development.json');ap.add_argument('--rerank',action='store_true')
    a=ap.parse_args();papers=json.loads(Path(a.corpus).read_text());queries=json.loads(Path(a.queries).read_text())
    validate_queries(queries,papers);engine=SearchEngine(papers,backend=a.backend,device=a.device)
    report=dict(corpus_size=len(papers),query_count=len(queries),k=a.k,backend=a.backend,
        warning='Ad hoc/development evaluation, not an independently held-out test.',provenance=provenance(engine,a.corpus),results=[])
    for mode,rerank in [('bm25',False),('semantic',False),('hybrid',False)]+([('hybrid',True)] if a.rerank else []):
        engine.query_vector.cache_clear()
        report['results'].append(evaluate(engine,queries,mode=mode,k=a.k,alpha=a.alpha,candidate_k=a.candidate_k,rerank=rerank))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()

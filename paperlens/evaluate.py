import argparse,json,math,time,pathlib
import numpy as np
from .engine import SearchEngine

def metrics(results,relevance,k):
    ids=[p['id'] for p in results[:k]]
    relevant={i for i,g in relevance.items() if g>0}
    recall=len(set(ids)&relevant)/len(relevant) if relevant else 0
    dcg=sum((2**relevance.get(i,0)-1)/math.log2(r+2) for r,i in enumerate(ids))
    ideal=sum((2**g-1)/math.log2(r+2) for r,g in enumerate(sorted(relevance.values(),reverse=True)[:k]))
    rr=next((1/(r+1) for r,i in enumerate(ids) if i in relevant),0)
    return recall,dcg/ideal if ideal else 0,rr

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',default='data/papers.json');ap.add_argument('--queries',default='data/queries.json');ap.add_argument('--backend',choices=['lsa','transformer'],default='lsa');ap.add_argument('--k',type=int,default=10);ap.add_argument('--output',default='reports/evaluation.json');ap.add_argument('--rerank',action='store_true');args=ap.parse_args()
    engine=SearchEngine(json.load(open(args.corpus)),args.backend);queries=json.load(open(args.queries));report={'corpus_size':len(engine.papers),'query_count':len(queries),'backend':args.backend,'k':args.k,'warning':'Curated development toy set; not an independent benchmark or generalization claim. Latency excludes indexing/model load.','results':[]}
    for mode in ('bm25','semantic','hybrid'):
        values=[];times=[]
        for q in queries:
            start=time.perf_counter();out=engine.search(q['text'],mode,k=args.k,rerank=args.rerank);times.append((time.perf_counter()-start)*1000);values.append(metrics(out,q['relevance'],args.k))
        avg=np.mean(values,axis=0);report['results'].append(dict(mode=mode,recall_at_k=float(avg[0]),ndcg_at_k=float(avg[1]),mrr_at_k=float(avg[2]),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95))))
    pathlib.Path(args.output).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__': main()

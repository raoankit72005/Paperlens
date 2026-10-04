"""Convert locally downloaded BEIR corpus.jsonl, queries.jsonl and qrels TSV."""
import argparse,json,csv,pathlib
ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('--split',default='test');ap.add_argument('--output',default='data/beir');a=ap.parse_args();root=pathlib.Path(a.directory);out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
corpus=[]
with (root/'corpus.jsonl').open() as f:
    for line in f:
        r=json.loads(line);corpus.append(dict(id=r['_id'],title=r.get('title',''),summary=r['text'],year=0,topic='Benchmark',url=''))
qrels={}
with (root/'qrels'/f'{a.split}.tsv').open() as f:
    for r in csv.DictReader(f,delimiter='\t'):
        if int(r['score'])>0:qrels.setdefault(r['query-id'],{})[r['corpus-id']]=int(r['score'])
queries=[]
with (root/'queries.jsonl').open() as f:
    for line in f:
        r=json.loads(line)
        if r['_id'] in qrels:queries.append(dict(id=r['_id'],text=r['text'],relevance=qrels[r['_id']]))
(out/'papers.json').write_text(json.dumps(corpus));(out/'queries.json').write_text(json.dumps(queries));print(f'Converted {len(corpus)} documents and {len(queries)} labeled queries')

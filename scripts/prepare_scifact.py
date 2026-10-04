"""Prepare full BEIR SciFact corpus, train-derived development queries and official test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
# Checksum published in the official BEIR dataset table.
MD5='5f7d1de60b170fc8027bb7898e2efca1'


def prepare(source, output, dev_limit=100):
    if dev_limit<1: raise ValueError('Development limit must be positive')
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    papers=[]
    with (source/'corpus.jsonl').open() as f:
        for line in f:
            r=json.loads(line);papers.append(dict(id=str(r['_id']),title=r.get('title',''),summary=r['text'],year=0,topic='SciFact',url=''))
    texts={}
    with (source/'queries.jsonl').open() as f:
        for line in f:
            r=json.loads(line);texts[str(r['_id'])]=r['text']
    def split(name):
        labels={}
        with (source/'qrels'/f'{name}.tsv').open() as f:
            for row in csv.DictReader(f,delimiter='\t'):
                if int(row['score'])>0: labels.setdefault(row['query-id'],{})[row['corpus-id']]=int(row['score'])
        return [dict(id=i,text=texts[i],relevance=labels[i]) for i in sorted(labels)]
    dev,test=split('train'),split('test')
    test_ids={q['id'] for q in test};test_texts={q['text'].strip().casefold() for q in test}
    dev=[q for q in dev if q['id'] not in test_ids and q['text'].strip().casefold() not in test_texts]
    # Stable hash order makes the development sample independent of relevance counts.
    dev=sorted(dev,key=lambda q:hashlib.sha256(('paperlens-dev-v1:'+q['id']).encode()).hexdigest())[:dev_limit]
    if not dev or not test: raise ValueError('Both splits must be nonempty')
    if len({p['id'] for p in papers})!=len(papers): raise ValueError('Duplicate corpus IDs')
    corpus_ids={p['id'] for p in papers}
    if any(not set(q['relevance']).issubset(corpus_ids) for q in dev+test): raise ValueError('Missing relevant corpus documents')
    files={'papers.json':papers,'dev_queries.json':dev,'test_queries.json':test}
    checksums={}
    for name,content in files.items():
        path=output/name;path.write_text(json.dumps(content,indent=2));checksums[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(dataset='BEIR/SciFact',source_url=URL,dev_source='official train qrels, deterministic hash sample',test_source='all official test qrels; never subsampled',dev_count=len(dev),test_count=len(test),corpus_count=len(papers),files=checksums)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',help='Existing extracted SciFact folder; skips download');ap.add_argument('--output',default='data/scifact');ap.add_argument('--dev-limit',type=int,default=100);a=ap.parse_args()
    if a.source: source=Path(a.source)
    else:
        cache=Path('.cache/benchmarks');cache.mkdir(parents=True,exist_ok=True);archive=cache/'scifact.zip'
        if not archive.exists():
            with urllib.request.urlopen(URL,timeout=60) as response,archive.with_suffix('.tmp').open('wb') as f:
                while chunk:=response.read(1024*1024): f.write(chunk)
            archive.with_suffix('.tmp').replace(archive)
        if hashlib.md5(archive.read_bytes()).hexdigest()!=MD5: raise ValueError('SciFact download checksum mismatch; remove cached zip and retry')
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target=(cache/name).resolve()
                if not target.is_relative_to(cache.resolve()): raise ValueError('Unsafe archive path')
            z.extractall(cache)
        source=cache/'scifact'
    print(json.dumps(prepare(source,a.output,a.dev_limit),indent=2))

if __name__=='__main__': main()

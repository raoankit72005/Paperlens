"""Validate integration contracts without downloading neural weights."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from paperlens.engine import SearchEngine

class Encoder:
    max_seq_length=256
    document_calls=0
    def __init__(self,*args,**kwargs): pass
    def get_sentence_embedding_dimension(self): return 2
    def encode(self,texts,**kwargs):
        if len(texts)>1: type(self).document_calls+=1
        return np.array([[1.,0.] if 'teacher' in t else [0.,1.] for t in texts],dtype=np.float32)

class Reranker:
    def __init__(self,*args,**kwargs): pass
    def predict(self,pairs,**kwargs): return [9. if 'second' in text else 1. for _,text in pairs]

class NeuralContractTest(unittest.TestCase):
    def setUp(self):
        self.papers=[dict(id='a',title='first teacher',summary='distillation',year=2020,topic='ML'),dict(id='b',title='second image',summary='vision',year=2021,topic='ML')]
        self.module=types.SimpleNamespace(SentenceTransformer=Encoder,CrossEncoder=Reranker)
    def test_cache_and_reranking(self):
        Encoder.document_calls=0
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{'sentence_transformers':self.module}):
            e=SearchEngine(self.papers,cache_dir=folder)
            self.assertEqual(e.search('teacher',mode='semantic',k=1,candidate_k=2)[0]['id'],'a')
            result=e.search('teacher',mode='semantic',k=1,candidate_k=2,rerank=True)
            self.assertEqual(result[0]['id'],'b');self.assertEqual(result[0]['score'],result[0]['rerank_score'])
            again=SearchEngine(self.papers,cache_dir=folder);self.assertTrue(again.cache_hit)
            self.assertEqual(Encoder.document_calls,1)
            changed=[dict(p) for p in self.papers];changed[0]['summary']='changed'
            self.assertFalse(SearchEngine(changed,cache_dir=folder).cache_hit)
    def test_bm25_does_not_encode(self):
        e=SearchEngine(self.papers,backend='bm25')
        with patch.object(e,'semantic',side_effect=AssertionError('unnecessary dense retrieval')):
            self.assertEqual(e.search('teacher',mode='bm25',k=1)[0]['id'],'a')
    def test_invalid_candidate_depth(self):
        e=SearchEngine(self.papers,backend='bm25')
        with self.assertRaises(ValueError):e.search('teacher',k=10,candidate_k=5)

if __name__=='__main__': unittest.main()

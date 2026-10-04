import unittest,json
from paperlens.engine import SearchEngine
from paperlens.evaluate import metrics
class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.e=SearchEngine(json.loads(__import__('pathlib').Path('data/papers.json').read_text()),backend='lsa')
    def test_exact_title(self): self.assertEqual(self.e.search('QLoRA',mode='bm25')[0]['id'],'2305.14314')
    def test_blank_and_unknown(self):
        self.assertEqual(self.e.search(''),[]);self.assertEqual(self.e.search('zzzzunseenword'),[])
    def test_filters(self): self.assertTrue(all(p['topic']=='Retrieval' and p['year']>=2021 for p in self.e.search('retrieval',topic='Retrieval',min_year=2021)))
    def test_fusion_endpoints(self):
        for a,m in [(0,'bm25'),(1,'semantic')]: self.assertEqual([p['id'] for p in self.e.search('attention memory',alpha=a)],[p['id'] for p in self.e.search('attention memory',mode=m)])
    def test_metrics(self): self.assertEqual(metrics([{'id':'a'}],{'a':2},10),(1,1,1))
if __name__=='__main__': unittest.main()

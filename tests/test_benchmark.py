import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from scripts.prepare_scifact import prepare
from paperlens.benchmark import validate_queries

class HeldOutTest(unittest.TestCase):
    def test_prepare_and_frozen_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();(source/'qrels').mkdir()
            docs=[{'_id':'a','title':'teacher model','text':'distillation compression'},{'_id':'b','title':'image patches','text':'vision transformer'},{'_id':'c','title':'document retrieval','text':'sparse semantic search'}]
            queries=[{'_id':'dev','text':'compress a teacher model'},{'_id':'test','text':'search documents'}]
            (source/'corpus.jsonl').write_text('\n'.join(map(json.dumps,docs)))
            (source/'queries.jsonl').write_text('\n'.join(map(json.dumps,queries)))
            (source/'qrels/train.tsv').write_text('query-id\tcorpus-id\tscore\ndev\ta\t1\n')
            (source/'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\ntest\tc\t1\n')
            data=root/'data';manifest=prepare(source,data);self.assertEqual(manifest['test_count'],1)
            config=root/'config.json'
            cmd=[sys.executable,'-m','paperlens.benchmark']
            subprocess.run(cmd+['tune','--data',str(data),'--config',str(config),'--output',str(root/'dev.json'),'--backend','lsa','--no-rerank','--alphas','.5','--candidate-counts','3','--k','2'],check=True,capture_output=True)
            frozen=config.read_bytes()
            subprocess.run(cmd+['test','--data',str(data),'--config',str(config),'--output',str(root/'test.json')],check=True,capture_output=True)
            self.assertEqual(config.read_bytes(),frozen)
            report=json.loads((root/'test.json').read_text());self.assertEqual(len(report['results']),3);self.assertEqual(report['query_count'],1)
            (data/'test_queries.json').write_text('[]')
            failed=subprocess.run(cmd+['test','--data',str(data),'--config',str(config)],capture_output=True)
            self.assertNotEqual(failed.returncode,0)
    def test_missing_relevant_document_rejected(self):
        with self.assertRaises(ValueError):validate_queries([dict(id='q',text='query',relevance={'missing':1})],[dict(id='a')])

if __name__=='__main__': unittest.main()

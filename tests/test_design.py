import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from rq1.config import load_settings
from rq1.data.graphrag_bench import load_novel
from rq1.experiments.cache import guard_run
from rq1.experiments.analysis import analyze
from rq1.chunking.fixed import Chunk
from rq1.eval.retrieval_metrics import evidence_metrics

ROOT=Path(__file__).resolve().parents[1]

class DesignTests(unittest.TestCase):
    def test_official_schema_and_missing_source(self):
        with tempfile.TemporaryDirectory() as d:
            c,q=Path(d)/'c.json',Path(d)/'q.json'
            c.write_text(json.dumps([dict(corpus_name='book',context='gold text')]))
            row=dict(id='q',source='book',question='Who?',answer='A',question_type='Fact Retrieval',
                     evidence='gold text',evidence_triple='(A, knows, B)')
            q.write_text(json.dumps([row]))
            docs,qs=load_novel(c,q)
            self.assertEqual(qs['book'][0].evidence,('gold text',))
            self.assertEqual(qs['book'][0].evidence_relations,('(A, knows, B)',))
            row['source']='absent'; q.write_text(json.dumps([row]))
            with self.assertRaises(ValueError): load_novel(c,q)

    def test_wrong_document_consumes_rank_and_budget(self):
        wrong=Chunk('w','wrong','gold',0,4,1)
        right=Chunk('r','right','gold',0,4,1)
        score=evidence_metrics(('gold',),[wrong,right],'right','gold',1)
        self.assertEqual(score['evidence_recall_at_1'],0)
        self.assertEqual(score['fixed_budget_evidence_recall'],0)
        self.assertEqual(score['evidence_span_precision'],0)

    def test_cache_reuse_and_invalidation(self):
        settings=load_settings(ROOT/'configs/synthetic.json')
        docs,qs=load_novel(settings.corpus, settings.questions)
        with tempfile.TemporaryDirectory() as d:
            s=replace(settings,output=Path(d))
            self.assertEqual(guard_run(s,docs,qs),guard_run(s,docs,qs))
            with self.assertRaises(ValueError): guard_run(replace(s,model='changed'),docs,qs)
            self.assertNotIn('api_key', (Path(d)/'cache_identity.json').read_text())

    def test_rq1_cluster_stability_and_incomplete_grid(self):
        rows=[dict(id=f'q{n}',source=f'b{n}',g_E=e,g_R=r,status='ok',answer_f1=.8 if e!=r else .5)
              for n in range(6) for e in (4,8) for r in (4,8)]
        a=analyze(rows,repetitions=50)
        self.assertEqual(a['global_best'],[4,8])
        self.assertAlmostEqual(a['bootstrap_off_diagonal_mass'],1.0)
        self.assertEqual(a,analyze(rows,repetitions=50))
        self.assertEqual(analyze(rows[:-1],repetitions=5)['status'],'incomplete')


class FullGridTests(unittest.TestCase):
    def test_full_64_cells_and_eight_graphs(self):
        from rq1.experiments.run_grid import run
        import csv
        with tempfile.TemporaryDirectory() as d:
            conf=json.loads((ROOT/'configs/synthetic.json').read_text())
            conf['experiment']['sizes']=[128,256,512,768,1024,1200,1536,2048]
            conf['data']['root']=str(ROOT/'examples/synthetic')
            conf['data']['output']=str(Path(d)/'out')
            p=Path(d)/'config.json';p.write_text(json.dumps(conf))
            out=run(p)
            manifest=json.loads((out/'run_manifest.json').read_text())
            self.assertEqual(manifest['completed_cells'],64)
            self.assertEqual(len(list((out/'graphs').glob('e*'))),8)
            self.assertTrue((out/'rq1_analysis.json').exists())
            with (out/'per_query_results.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),128)

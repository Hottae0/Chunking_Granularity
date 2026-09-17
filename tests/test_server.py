import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from rq1.config import load_settings
from rq1.server import preflight, serve_command
from rq1.msgraphrag.indexer import REQUIRED_TABLES, _input_name, _patch_settings
from rq1.experiments.cache import guard_run
from rq1.experiments.prepare_subset import _validate_graph
from rq1.data.graphrag_bench import load_novel, relation_statements

ROOT=Path(__file__).resolve().parents[1]

class ServerTests(unittest.TestCase):
    def settings(self):
        return replace(load_settings(ROOT/'configs/synthetic.json'),backend='official',model='chat',
                       embedding_model='embed',embedding_base_url='http://localhost:8001/v1',
                       embedding_api_key='private-embedding-key',embedding_dimensions=3)

    def test_model_command_preserves_allocation(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-allocated','VLLM_BIN':'/opt/vllm/bin/vllm'}):
            command=serve_command('embedding',self.settings(),memory=.6,max_model_len=512)
            self.assertEqual(command[0],'/opt/vllm/bin/vllm')
            self.assertIn('pooling',command)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'GPU-allocated')
        with self.assertRaises(ValueError): serve_command('chat',self.settings(),tensor_parallel=0)

    def test_preflight_two_endpoints_and_dimensions(self):
        calls=[]
        class Client:
            def __init__(self,**kw):
                calls.append(kw)
                self.models=NS(list=lambda:NS(data=[NS(id='chat'),NS(id='embed')]))
                self.embeddings=NS(create=lambda **kw:NS(data=[NS(embedding=[.1,.2,.3])]))
                self.chat=NS(completions=NS(create=lambda **kw:NS(choices=[NS(message=NS(content='{"ok":true}'))])))
            def close(self): pass
        self.assertTrue(preflight(self.settings(),factory=Client)['structured_output'])
        self.assertEqual(calls[1]['base_url'],'http://localhost:8001/v1')
        with self.assertRaisesRegex(ValueError,'dimensions mismatch'):
            preflight(replace(self.settings(),embedding_dimensions=4),factory=Client)

    def test_official_nested_relation_annotations(self):
        expected = ("(Alice, knows, Bob)", "(Bob, visits, Cornwall)")
        self.assertEqual(relation_statements([
            ["Alice", "knows", "Bob"],
            ["Bob", "visits", "Cornwall"],
        ]), expected)
        self.assertEqual(
            relation_statements("[['Alice', 'knows', 'Bob'], ['Bob', 'visits', 'Cornwall']]"),
            expected,
        )
        mapping = relation_statements({"source": "Alice", "relation": "knows", "target": "Bob"})
        self.assertEqual(len(mapping), 1)
        self.assertIn('"source": "Alice"', mapping[0])

    def test_subset_graph_cache_validation(self):
        import yaml
        settings = self.settings()
        documents, _ = load_novel(settings.corpus, settings.questions)
        with tempfile.TemporaryDirectory() as tmp:
            graph = Path(tmp)
            (graph / 'input').mkdir()
            (graph / 'output').mkdir()
            titles = {}
            for doc in documents:
                name = _input_name(doc.name)
                titles[name] = doc.name
                (graph / 'input' / name).write_text(doc.text, encoding='utf-8')
            (graph / 'input_titles.json').write_text(json.dumps(titles), encoding='utf-8')
            raw = {
                'completion_models': {'chat': {'model': settings.model}},
                'embedding_models': {'embed': {'model': settings.embedding_model}},
                'chunking': {'size': 256, 'overlap': 0},
                'vector_store': {'vector_size': settings.embedding_dimensions},
            }
            (graph / 'settings.yaml').write_text(yaml.safe_dump(raw), encoding='utf-8')
            self.assertFalse(_validate_graph(graph, 256, documents, settings))
            for table in REQUIRED_TABLES:
                (graph / 'output' / f'{table}.parquet').touch()
            (graph / 'graph_complete.json').write_text('{}', encoding='utf-8')
            self.assertTrue(_validate_graph(graph, 256, documents, settings))
            with self.assertRaisesRegex(ValueError, 'chat model differs'):
                _validate_graph(graph, 256, documents, replace(settings, model='other'))

    def test_index_settings_and_secret_free_cache(self):
        import yaml
        settings=self.settings()
        raw=dict(completion_models={'chat':{}},embedding_models={'embed':{}},chunking={},
                 local_search={},cluster_graph={},output_storage={},cache={'storage':{}},
                 vector_store={'index_schema':{'entity':{'vector_size':3072}}})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'settings.yaml';path.write_text(yaml.safe_dump(raw))
            _patch_settings(path,settings,128)
            result=yaml.safe_load(path.read_text())
            self.assertEqual(result['embedding_models']['embed']['api_base'],settings.embedding_base_url)
            self.assertEqual(result['embedding_models']['embed']['api_key'],'${GRAPHRAG_EMBEDDING_API_KEY}')
            self.assertEqual(result['vector_store']['vector_size'],3)
            self.assertEqual(result['vector_store']['index_schema']['entity']['vector_size'],3)
            docs,qs=load_novel(settings.corpus,settings.questions)
            guard_run(replace(settings,output=Path(tmp)),docs,qs)
            self.assertNotIn(settings.embedding_api_key,(Path(tmp)/'cache_identity.json').read_text())

"""Offline regression tests for batching and preparation recovery."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from post_training import judge, data
from post_training.core import file_hash, digest
from test_post_judge import sample, config, response


class BatchRepairTests(unittest.TestCase):
    def setUp(self):
        root = judge.ROOT / 'post_training_runs'
        root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.input = self.path / 'samples.jsonl'
        self.output = self.path / 'scores.jsonl'
        self.rows = [sample('s' + str(i)) for i in range(4)]
        self.input.write_text(''.join(json.dumps(x) + '\n' for x in self.rows), encoding='utf-8')
        self.cfg = config()
        self.cfg['api'].update(batch_size=8, thinking={'type': 'disabled'})
        self.calls = []

    def transport(self, endpoint, payload, key, timeout):
        self.calls.append(payload)
        items = json.loads(payload['messages'][1]['content'])['items']
        result = {'items': [{'item_id': x['item_id'], 'result': {
            'scores': {name: 4 for name in judge.result_schema('score')['properties']['scores']['properties']},
            'reason': '清楚。'}} for x in reversed(items)]}
        return response(result)

    def run_judge(self, **kwargs):
        return judge.run_judging(self.cfg, self.input, self.output, 'score', execute=True,
            transport=kwargs.pop('transport', self.transport), environ={'POSTTRAIN_API_KEY': 'test-secret'},
            sleep=lambda _: None, **kwargs)

    def test_batch_maps_shuffled_ids_and_resume_makes_no_requests(self):
        summary = self.run_judge()
        self.assertEqual(summary['physical_requests'], 1)
        self.assertEqual(summary['remaining_items'], 0)
        labels = [json.loads(x) for x in self.output.read_text(encoding='utf-8').splitlines()]
        self.assertEqual({x['item_id'] for x in labels}, {x['sample_id'] for x in self.rows})
        self.assertEqual({x['item_hash'] for x in labels}, {digest(x) for x in self.rows})
        self.run_judge()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]['thinking'], {'type': 'disabled'})

    def test_physical_cap_counts_batches_not_items(self):
        self.cfg['api'].update(batch_size=2, max_requests=1)
        summary = self.run_judge()
        self.assertEqual(summary['successful_items'], 2)
        self.assertEqual(summary['remaining_items'], 2)
        self.assertEqual(summary['physical_requests'], 1)

    def test_batch_never_combines_different_tasks(self):
        self.rows[2].update(task_id='t2', group_id='g2')
        self.rows[3].update(task_id='t2', group_id='g2')
        self.input.write_text(''.join(json.dumps(x) + '\n' for x in self.rows), encoding='utf-8')
        summary = self.run_judge()
        self.assertEqual(summary['physical_requests'], 2)

    def test_invalid_batch_splits_without_saving_wrong_ids(self):
        def transport(endpoint, payload, key, timeout):
            body = json.loads(payload['messages'][1]['content'])
            if len(body['items']) > 1:
                self.calls.append(payload)
                return response({'items': []})
            return self.transport(endpoint, payload, key, timeout)
        self.cfg['api']['max_requests'] = 20
        summary = self.run_judge(transport=transport)
        self.assertEqual(summary['successful_items'], 4)
        self.assertEqual(summary['physical_requests'], 7)
        self.assertEqual(summary['remaining_items'], 0)

    def test_thinking_and_batch_size_are_bound_to_cache(self):
        self.run_judge()
        for change in ({'batch_size': 4}, {'thinking': {'type': 'enabled'}}):
            cfg = copy.deepcopy(self.cfg)
            cfg['api'].update(change)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                judge.run_judging(cfg, self.input, self.output, 'score')

    def test_invalid_thinking_is_rejected_before_request(self):
        self.cfg['api']['thinking'] = {'type': 'maybe'}
        with self.assertRaises(ValueError):
            self.run_judge()
        self.assertEqual(self.calls, [])


class PreparationRepairTests(unittest.TestCase):
    def test_missing_manifest_has_recovery_instruction_without_torch(self):
        from post_training.prepared import load_prepared
        with tempfile.TemporaryDirectory(dir=judge.ROOT / 'post_training_runs') as tmp:
            with self.assertRaisesRegex(ValueError, 'data prepare'):
                load_prepared(tmp, 'sft')

    def test_summary_reports_empty_and_missing_types(self):
        from post_training.prepared import readiness
        result = readiness({'train': [sample()], 'val': []})
        self.assertFalse(result['ready'])
        self.assertEqual(len(result['missing']['val']), 4)
        self.assertIn('free/qijue', result['missing']['train'])

    def test_prepare_is_idempotent_and_rejects_changed_inputs(self):
        from post_training.core import atomic_json
        root = judge.ROOT / 'post_training_runs'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            base = Path(tmp)
            source = base / 'samples.jsonl'
            scores = base / 'scores.jsonl'
            rows = [sample()]
            source.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
            atomic_json(base / 'manifest.json', {'status':'completed', 'samples_sha256':file_hash(source),
                'sample_count':1, 'expected_count':1})
            label = {'item_id':'s1', 'item_hash':digest(rows[0]), 'status':'ok', 'judge_id':'legacy',
                'result':{'scores':dict.fromkeys(judge.result_schema('score')['properties']['scores']['properties'], 4), 'reason':'ok'}}
            scores.write_text(json.dumps(label)+'\n', encoding='utf-8')
            output = base / 'prepared'
            args = ['prepare','--samples',str(source),'--scores',str(scores),'--output',str(output)]
            data.main(args)
            before = (output/'manifest.json').read_bytes()
            data.main(args)
            self.assertEqual((output/'manifest.json').read_bytes(), before)
            scores.write_text(json.dumps(dict(label,judge_id='changed'))+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '.*'):
                data.main(args)

    def test_pipeline_dry_run_does_not_create_prepared_or_train(self):
        from post_training import pipeline
        from post_training.core import atomic_json
        root = judge.ROOT / 'post_training_runs'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            base = Path(tmp)
            source = base/'samples.jsonl'
            source.write_text(json.dumps(sample())+'\n',encoding='utf-8')
            atomic_json(base/'manifest.json',{'status':'completed','samples_sha256':file_hash(source),
                'sample_count':1,'expected_count':1})
            out = base/'prepared'
            with patch('post_training.judge.http_transport',side_effect=AssertionError('network')):
                self.assertEqual(pipeline.main(['--candidates',str(base),'--prepared',str(out)]),0)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()

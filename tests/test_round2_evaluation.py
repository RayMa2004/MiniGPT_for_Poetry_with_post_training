"""Offline regression tests for the second-round evaluator."""
import importlib.util
import json
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('round2_eval',ROOT/'scripts'/'round2_eval.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Round2EvaluationTests(unittest.TestCase):
    def test_quality_reports_all_four_dimensions_and_threshold(self):
        score={'fluency':4,'coherence':3,'imagery':4,'poeticness':3}
        samples=[{'sample_id':'s1','mode':'free','form':'wujue'}]
        labels=[{'item_id':'s1','item_hash':module.digest(samples[0]),'judge_id':'j',
                 'status':'ok','result':{'scores':score,'reason':'ok'}}]
        result=module.quality_summary(samples,labels,{'minimum_mean':3.5,'minimum_dimension':3})
        self.assertEqual(result['count'],1)
        self.assertEqual(result['qualified_count'],1)
        self.assertEqual(result['dimension_mean']['fluency'],4)
        self.assertEqual(result['by_group']['free/wujue']['count'],1)

    def test_quality_rejects_incomplete_labels(self):
        with self.assertRaises(ValueError):
            module.quality_summary([{'sample_id':'missing','mode':'free','form':'wujue'}],[],
                                   {'minimum_mean':3.5,'minimum_dimension':3})

    def test_idempotent_jsonl_refuses_changed_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path=Path(tmp)/'rows.jsonl'
            module.ensure_jsonl(path,[{'sample_id':'a'}])
            module.ensure_jsonl(path,[{'sample_id':'a'}])
            with self.assertRaises(ValueError):
                module.ensure_jsonl(path,[{'sample_id':'b'}])
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['sample_id'],'a')

    def test_plan_does_not_read_api_key_or_call_network(self):
        with patch.object(module.judge,'run_judging',side_effect=AssertionError('network or cache')):
            self.assertEqual(module.main(['--split','val']),0)

    def test_actual_workflow_resumes_after_failure_and_reuses_completed_results(self):
        calls=[]
        def transport(endpoint,payload,key,timeout):
            calls.append(1)
            if len(calls)==2:
                raise module.judge.ProviderError('http_error',status=401)
            items=json.loads(payload['messages'][1]['content'])['items']
            results=[]
            for item in items:
                if item['task']=='score':
                    result={'scores':dict.fromkeys(module.DIMENSIONS,4),'reason':'ok'}
                else:
                    result={'winner':'A','reason':'ok'}
                results.append({'item_id':item['item_id'],'result':result})
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'items':results})}}],
                    'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}}
        parent=ROOT/'post_training_data'
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            out=Path(tmp)/'eval'
            args=['--split','val','--output',str(out),'--execute']
            with patch.dict(os.environ,{'POSTTRAIN_API_KEY':'offline-test-secret'}), \
                 patch.object(module.judge,'http_transport',side_effect=transport), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(args),1)
                cached=module.read_jsonl(out/'base_vs_round2_rsft'/'preferences.jsonl')
                self.assertGreater(len(cached),0)
                self.assertEqual(module.main(args),0)
                self.assertEqual(module.main(args),0)
            report=module.read_json(out/'quality_summary.json')
            self.assertEqual(report['models']['baseline']['count'],96)
            self.assertEqual(report['models']['round2_ppo']['qualified_count'],96)
            self.assertEqual(module.read_json(out/'base_vs_round2_ppo'/'report.json')['matched_count'],96)


if __name__=='__main__':
    unittest.main()

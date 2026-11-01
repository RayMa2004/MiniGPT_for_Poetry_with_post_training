"""Offline tests for reward-guided candidate selection."""
import unittest
from post_training.bestof import select_candidates
from test_post_data import sample


class BestOfTests(unittest.TestCase):
    def test_picks_highest_score_for_each_task(self):
        a=sample(0)
        b=sample(1,text="春山风月夜，江水入云深。花落无人见，清光照故林。")
        result=select_candidates([a,b],{'sample-0':0.2,'sample-1':0.9},1)
        self.assertEqual([r['sample_id'] for r in result['train']],['sample-1'])
        self.assertEqual(result['train'][0]['reward_score'],0.9)

    def test_test_text_cannot_leak_to_train(self):
        a=sample(0)
        b=sample(1,split='test',prompt='')
        b['group_id']='free:heldout'
        selected=select_candidates([a,b],{'sample-0':2.,'sample-1':1.},1)
        self.assertEqual(selected['train'],[])
        self.assertEqual(len(selected['test']),1)

    def test_missing_and_nonfinite_scores_rejected(self):
        for scores in ({}, {'sample-0':float('nan')}, {'sample-0':True}, {'sample-0':1.,'extra':2.}):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select_candidates([sample()],scores,1)

    def test_duplicate_texts_do_not_consume_top_k(self):
        a=sample(0); b=sample(1); c=sample(2,text="春山风月夜，江水入云深。花落无人见，清光照故林。")
        selected=select_candidates([a,b,c],{'sample-0':3.,'sample-1':2.,'sample-2':1.},2)
        self.assertEqual({x['sample_id'] for x in selected['train']},{'sample-0','sample-2'})

if __name__ == '__main__':
    unittest.main()

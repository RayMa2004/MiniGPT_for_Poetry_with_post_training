"""Remote-only evaluation data tests (not executed locally)."""
import unittest
from post_training.core import digest
from post_training.compare import comparison_pairs, summarize_comparison


def datasets():
    baseline=[]
    candidate=[]
    for mode in ("prefix","free"):
        for form in ("wujue","qijue"):
            prompt="春" if mode == "prefix" else ""
            text=("春山风月夜，江水入云深。花落无人见，清光照客心。" if form == "wujue" else
                  "春山风月照清夜，江水悠悠入梦深。花落无声人不见，清光一片照客心。")
            task=mode+form
            a={"task_id":task,"group_id":"prefix:春" if prompt else "free:test","prompt":prompt,
               "form":form,"mode":mode,"split":"test","sample_id":"base-"+task,"text":text,
               "continuation":text[len(prompt):],"sample_index":0,"seed":42,"checkpoint_sha256":"a"*64,
               "vocab_sha256":"b"*64,"sampling":{"repetition_penalty":1.1}}
            b=dict(a,sample_id="new-"+task,checkpoint_sha256="c"*64,text=text.replace("客心","故林"))
            b["continuation"]=b["text"][len(prompt):]
            baseline.append(a)
            candidate.append(b)
    return baseline,candidate


def labels_for(comparison):
    return [{"item_id":p["pair_id"],"item_hash":digest(p),"judge_id":"judge-v1","status":"ok",
             "result":{"winner":p["candidate_side"],"reason":"better"}} for p in comparison["pairs"]]


class ComparisonTests(unittest.TestCase):
    def test_identical_text_is_local_tie_no_paid_comparison(self):
        left,right=datasets()
        right=[dict(b,text=a["text"],continuation=a["continuation"]) for a,b in zip(left,right)]
        result=comparison_pairs(left,right,42,"test")
        self.assertEqual(result["pairs"],[])
        self.assertEqual(len(result["identical"]),4)
        self.assertEqual(summarize_comparison(result,[])["overall"]["identical"],4)

    def test_candidate_side_remains_correct_after_blinding(self):
        left,right=datasets()
        result=comparison_pairs(left,right,42,"test")
        report=summarize_comparison(result,labels_for(result))
        self.assertEqual(report["overall"]["candidate_wins"],4)
        self.assertEqual(report["overall"]["baseline_wins"],0)

    def test_missing_judgment_never_yields_complete_report(self):
        left,right=datasets()
        with self.assertRaises(ValueError):
            summarize_comparison(comparison_pairs(left,right,42,"test"),[])

    def test_missing_task_class_never_yields_complete_report(self):
        left,right=datasets()
        result=comparison_pairs(left[:1],right[:1],42,"test")
        with self.assertRaisesRegex(ValueError,"coverage"):
            summarize_comparison(result,labels_for(result))


if __name__ == "__main__":
    unittest.main()

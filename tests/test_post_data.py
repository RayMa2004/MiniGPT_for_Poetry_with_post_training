"""Remote-only contract tests; not executed during local development."""
import copy
import tempfile
import unittest
from pathlib import Path
from post_training.core import digest, write_jsonl, read_jsonl, validate_sample
from post_training.data import build_tasks, build_pairs, prepare


def sample(i=0, split="train", prompt="春", text="春山风月夜，江水入云深。花落无人见，清光照客心。"):
    return {"task_id": "task-"+split, "group_id": "prefix:"+prompt, "prompt": prompt,
            "form": "wujue", "mode": "prefix" if prompt else "free", "split": split,
            "sample_id": "sample-"+str(i), "text": text, "continuation": text[len(prompt):],
            "sample_index": i, "seed": i, "checkpoint_sha256": "a"*64,
            "vocab_sha256": "b"*64, "sampling": {"repetition_penalty": 1.1}}


def label(item, item_id, result):
    return {"item_id": item_id, "item_hash": digest(item), "judge_id": "judge-v1",
            "status": "ok", "result": result}


class PostDataTests(unittest.TestCase):
    def test_prefix_group_never_crosses_splits_and_both_forms_present(self):
        tasks = build_tasks(["春"+ch for ch in "山风花月夜江水云清光落客心天地日雨雪梅竹"], 10, 42)
        mapping = {}
        for task in tasks:
            mapping.setdefault(task["group_id"], set()).add(task["split"])
        self.assertTrue(all(len(v) == 1 for v in mapping.values()))
        for split in ("train", "val", "test"):
            self.assertEqual({(t["mode"], t["form"]) for t in tasks if t["split"] == split},
                             {(m, f) for m in ("prefix", "free") for f in ("wujue", "qijue")})
        self.assertEqual(tasks, build_tasks(["春"+ch for ch in "山风花月夜江水云清光落客心天地日雨雪梅竹"], 10, 42))

    def test_calibration_and_formal_empty_pools_have_disjoint_task_ids(self):
        prefixes=["春"+ch for ch in "山风花月夜江水云清光落客心天地日雨雪梅竹"]
        pilot=build_tasks(prefixes,10,42,namespace="pilot")
        formal=build_tasks(prefixes,10,42,namespace="main")
        self.assertTrue({t["task_id"] for t in pilot}.isdisjoint({t["task_id"] for t in formal}))
        self.assertTrue({t["group_id"] for t in pilot if t["mode"] == "free"}.isdisjoint(
                        {t["group_id"] for t in formal if t["mode"] == "free"}))

    def test_bad_prefix_or_duplicate_id_is_rejected(self):
        bad = sample()
        bad["continuation"] = bad["text"]
        with self.assertRaises(ValueError):
            validate_sample(bad)
        bad = sample()
        bad["text"] = "不合诗体"
        with self.assertRaises(ValueError):
            validate_sample(bad)

    def test_jsonl_never_overwrites_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"records.jsonl"
            write_jsonl(path, [{"text": "春"}])
            with self.assertRaises(FileExistsError):
                write_jsonl(path, [])
            self.assertEqual(read_jsonl(path), [{"text": "春"}])

    def test_pairs_never_mix_tasks_or_identical_texts(self):
        rows = [sample(0), sample(1), sample(2, text="春山风月夜，江水入云深。花落无人见，清光照故林。"),
                sample(3, split="test", prompt="春山", text="春山风月夜，江水入云深。花落无人见，清光照故林。")]
        pairs = build_pairs(rows, 4, 42)
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertEqual(pair["a"]["task_id"], pair["b"]["task_id"])
            self.assertNotEqual(pair["a"]["text"], pair["b"]["text"])

    def test_selection_rejects_low_quality_and_stale_labels(self):
        rows = [sample(0), sample(1, text="春山风月夜，江水入云深。花落无人见，清光照故林。")]
        labels = [label(s, s["sample_id"], {"scores": dict.fromkeys(("fluency", "coherence", "imagery", "poeticness"), score), "reason": "brief"}) for s, score in zip(rows, (4, 2))]
        cfg = {"minimum_mean": 3.5, "minimum_dimension": 3, "max_per_task": 2}
        result = prepare(rows, labels, [], [], cfg)
        self.assertEqual([s["sample_id"] for s in result["sft"]["train"]], [rows[0]["sample_id"]])
        labels[0]["item_hash"] = "stale"
        with self.assertRaises(ValueError):
            prepare(rows, labels, [], [], cfg)

    def test_test_text_never_returned_to_sft_train(self):
        a = sample(0)
        b = sample(1, split="test", prompt="")
        b["group_id"] = "free:heldout"
        labels = [label(s, s["sample_id"], {"scores": dict.fromkeys(("fluency", "coherence", "imagery", "poeticness"), 4), "reason": "brief"}) for s in (a,b)]
        result = prepare([a,b], labels, [], [], {"minimum_mean":3.5,"minimum_dimension":3,"max_per_task":2})
        self.assertEqual(result["sft"]["train"], [])
        self.assertEqual(len(result["sft"]["test"]), 1)

    def test_pair_ties_do_not_create_preference(self):
        rows=[sample(0),sample(1,text="春山风月夜，江水入云深。花落无人见，清光照故林。")]
        pairs=build_pairs(rows,1,42)
        labels=[label(p,p["pair_id"],{"winner":"tie","reason":"similar"}) for p in pairs]
        result=prepare(rows,[],pairs,labels,{"minimum_mean":3.5,"minimum_dimension":3,"max_per_task":2})
        self.assertEqual(result["preferences"]["train"],[])
        labels[0]["result"]["winner"]="A"
        result=prepare(rows,[],pairs,labels,{"minimum_mean":3.5,"minimum_dimension":3,"max_per_task":2})
        self.assertEqual(result["preferences"]["train"][0]["chosen"]["sample_id"],pairs[0]["a"]["sample_id"])


if __name__ == "__main__":
    unittest.main()

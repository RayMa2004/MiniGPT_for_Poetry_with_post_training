import json
import tempfile
import unittest
from pathlib import Path

from experiment import Experiment, fingerprint, source_identity


class ExperimentTests(unittest.TestCase):
    def test_training_fingerprint_excludes_reporting_but_keeps_validation(self):
        scoped = {name.replace("\\", "/") for name in source_identity("training")}
        self.assertIn("model.py", scoped)
        self.assertIn("evaluation/metrics.py", scoped)
        self.assertIn("evaluation/__init__.py", scoped)
        self.assertNotIn("compare_experiments.py", scoped)
        self.assertNotIn("evaluation/plots.py", scoped)
        complete = {name.replace("\\", "/") for name in source_identity()}
        self.assertTrue({"project_paths.py", "evaluate.py", "generate.py"} <= complete)
        with self.assertRaises(ValueError):
            source_identity("unknown")

    def test_conflict_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Experiment.create(tmp, "E01_cosine", 42, {"seed": 42})
            with self.assertRaises(FileExistsError):
                Experiment.create(tmp, "E01_cosine", 42, {})
            run.log_metrics({"global_step": 1, "train_loss": 2})
            run.log_metrics({"global_step": 1, "val_loss": 3})
            self.assertEqual(len(run.read_metrics()), 1)
            self.assertEqual(run.read_metrics()[0]["train_loss"], "2")
            run.log_metrics({"global_step": 2})
            run.reconcile_metrics(1)
            self.assertEqual(len(run.read_metrics()), 1)
            resumed = Experiment.open(run.path)
            self.assertEqual(resumed.metadata["run_id"], "E01_cosine_seed42")

    def test_lock_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Experiment.create(tmp, "E00_baseline", 42, {})
            with run.lock():
                with self.assertRaises(FileExistsError):
                    with run.lock():
                        pass
            path = Path(tmp) / "a.txt"
            path.write_text("a", encoding="utf-8")
            first = fingerprint(path)["sha256"]
            path.write_text("b", encoding="utf-8")
            self.assertNotEqual(first, fingerprint(path)["sha256"])

    def test_no_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Experiment.create(tmp, "E00_baseline", 42, {})
            run.update(best_val_loss=float("inf"))
            self.assertIsNone(json.loads((run.path / "metadata.json").read_text(encoding="utf-8"))["best_val_loss"])

    def test_evaluation_id_is_namespaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Experiment.create(tmp, "natural", 42, {}, kind="evaluation", parent_run_id="E01_cosine_seed42")
            self.assertEqual(run.path.name, "natural_seed42")
            self.assertEqual(run.metadata["run_id"], "E01_cosine_seed42__natural_seed42")

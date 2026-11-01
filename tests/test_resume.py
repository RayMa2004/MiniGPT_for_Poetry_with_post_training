"""仅供远程运行：真实小模型连续训练与 Epoch 边界恢复的状态对照。"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train
from checkpoint import read_checkpoint
from config import load_config
from experiment import Experiment, data_identity, source_identity
from tokenizer import CharTokenizer


class ResumeTests(unittest.TestCase):
    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_epoch_resume_model_optimizer_rng_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poems = "春风秋月。\n山高水长。\n江南春雨。\n明月清风。\n"
            for name in ("train", "val"):
                (root / f"{name}.txt").write_text(poems, encoding="utf-8")
            CharTokenizer.build([poems]).save(root / "vocab.json")
            cfg = load_config()
            cfg["model"].update(n_embd=8, n_head=2, n_layer=1, block_size=16, dropout=0.1)
            cfg["training"].update(batch_size=2, max_epochs=2, scheduler="cosine", warmup_steps=1,
                                   log_every=1, deterministic=True)
            cfg["data"].update(train=str(root / "train.txt"), val=str(root / "val.txt"), vocab=str(root / "vocab.json"))
            cfg["data_identity"] = data_identity(cfg["data"])
            cfg["source_identity"] = source_identity()
            full = Experiment.create(root, "full", 42, cfg)
            split = Experiment.create(root, "split", 42, copy.deepcopy(cfg))
            train.train_run(full, cfg, torch.device("cpu"))
            original_save = train.save_checkpoint

            def interrupt_after_boundary(path, checkpoint):
                original_save(path, checkpoint)
                if Path(path).name == "last.pt" and checkpoint["epoch"] == 1:
                    raise KeyboardInterrupt("模拟边界提交后中断")

            with patch.object(train, "save_checkpoint", side_effect=interrupt_after_boundary):
                with self.assertRaises(KeyboardInterrupt):
                    train.train_run(split, cfg, torch.device("cpu"))
            train.train_run(split, cfg, torch.device("cpu"), resume=split.checkpoint_path())
            left, right = read_checkpoint(full.checkpoint_path()), read_checkpoint(split.checkpoint_path())
            for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state",
                        "global_step", "tokens_seen", "best_val_loss", "best_step"):
                self.assert_nested_equal(left[key], right[key])
            fields = ("global_step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "lr")
            self.assertEqual([{k: r.get(k) for k in fields} for r in full.read_metrics()],
                             [{k: r.get(k) for k in fields} for r in split.read_metrics()])
            events = EventAccumulator(str(split.path / "tensorboard")).Reload()
            steps = [event.step for event in events.Scalars("Loss/train")]
            self.assertEqual(steps, sorted(set(steps)))

    def test_metrics_and_events_after_boundary_are_purged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poems = "春风秋月。\n山高水长。\n江南春雨。\n明月清风。\n"
            for name in ("train", "val"):
                (root / f"{name}.txt").write_text(poems, encoding="utf-8")
            CharTokenizer.build([poems]).save(root / "vocab.json")
            cfg = load_config()
            cfg["model"].update(n_embd=8, n_head=2, n_layer=1, block_size=16, dropout=0.1)
            cfg["training"].update(batch_size=2, max_epochs=2, log_every=1, deterministic=True)
            cfg["data"].update(train=str(root / "train.txt"), val=str(root / "val.txt"), vocab=str(root / "vocab.json"))
            cfg["data_identity"], cfg["source_identity"] = data_identity(cfg["data"]), source_identity()
            run = Experiment.create(root, "purge", 42, cfg)
            original_evaluate = train.language_model_metrics
            calls = 0

            def interrupt_before_second_validation(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt("第二 Epoch 日志已落盘但尚未提交 Checkpoint")
                return original_evaluate(*args, **kwargs)

            with patch.object(train, "language_model_metrics", side_effect=interrupt_before_second_validation):
                with self.assertRaises(KeyboardInterrupt):
                    train.train_run(run, cfg, torch.device("cpu"))
            checkpoint = read_checkpoint(run.checkpoint_path())
            self.assertGreater(int(run.read_metrics()[-1]["global_step"]), checkpoint["global_step"])
            train.train_run(run, cfg, torch.device("cpu"), resume=run.checkpoint_path())
            rows = run.read_metrics()
            self.assertEqual(len(rows), 4)
            events = EventAccumulator(str(run.path / "tensorboard")).Reload()
            self.assertEqual([event.step for event in events.Scalars("Loss/train")], [1, 2, 3, 4])

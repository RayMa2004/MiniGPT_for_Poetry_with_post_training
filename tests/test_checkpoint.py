import random
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from checkpoint import capture_rng, restore_rng, read_checkpoint, load_model_checkpoint, save_checkpoint, resolve_model_config
from model import MiniGPT
from tokenizer import CharTokenizer


class CheckpointTests(unittest.TestCase):
    def test_constructor_defaults_not_training_defaults(self):
        cfg = resolve_model_config({"vocab_size": 10})
        self.assertEqual((cfg["n_embd"], cfg["n_head"], cfg["n_layer"]), (256, 4, 4))

    @unittest.skipUnless(os.environ.get("MINIGPT_LEGACY_CHECKPOINT") and os.environ.get("MINIGPT_LEGACY_VOCAB"),
                         "远程设置实际旧权重及词表路径后执行真实兼容测试")
    def test_real_legacy_artifact(self):
        path = os.environ["MINIGPT_LEGACY_CHECKPOINT"]
        vocab = os.environ["MINIGPT_LEGACY_VOCAB"]
        tokenizer = CharTokenizer.load(vocab)
        model, checkpoint = load_model_checkpoint(path, tokenizer, "cpu", vocab_path=vocab)
        self.assertEqual(model.vocab_size, tokenizer.vocab_size)
        self.assertIn("model_state_dict", checkpoint)

    def test_independent_legacy_key_fixture(self):
        # 不通过新模型生成旧权重，独立枚举旧实现的持久键名和形状。
        vocab, block, c, hidden = 6, 8, 8, 32
        state = {"token_embedding.weight": torch.randn(vocab, c),
                 "postition_embedding.weight": torch.randn(block, c),
                 "lm_head.weight": None, "ln_f.weight": torch.ones(c), "ln_f.bias": torch.zeros(c)}
        state["lm_head.weight"] = state["token_embedding.weight"]
        for name in ("ln1", "ln2"):
            state[f"blocks.0.{name}.weight"] = torch.ones(c)
            state[f"blocks.0.{name}.bias"] = torch.zeros(c)
        for name, out_dim, in_dim in (("attn.qkv", 3*c, c), ("attn.proj", c, c),
                                       ("ffn.net.0", hidden, c), ("ffn.net.2", c, hidden)):
            state[f"blocks.0.{name}.weight"] = torch.randn(out_dim, in_dim)
            state[f"blocks.0.{name}.bias"] = torch.zeros(out_dim)
        state["blocks.0.attn.causal_mask"] = torch.tril(torch.ones(block, block)).view(1, 1, block, block)
        model = MiniGPT(vocab, block_size=block, n_embd=c, n_head=2, n_layer=1)
        model.load_state_dict(state, strict=True)

    def test_rng_roundtrip(self):
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(1))
        restore_rng(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(1), expected[2])

    def test_old_and_new_strict_loading(self):
        tokenizer = CharTokenizer.build(["春风"])
        cfg = dict(vocab_size=tokenizer.vocab_size, n_embd=8, n_head=2, n_layer=1)
        model = MiniGPT(**cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save({"model_config": cfg, "model_state_dict": model.state_dict(), "epoch": 1}, path)
            loaded, checkpoint = load_model_checkpoint(path, tokenizer, "cpu")
            self.assertEqual(checkpoint.get("format_version", 1), 1)
            loaded.load_state_dict(model.state_dict(), strict=True)
            save_checkpoint(Path(tmp) / "new.pt", {"format_version": 2, "rng_state": capture_rng()})
            self.assertEqual(read_checkpoint(Path(tmp) / "new.pt")["format_version"], 2)

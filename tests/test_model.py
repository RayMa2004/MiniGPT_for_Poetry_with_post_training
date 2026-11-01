"""仅远程执行：本文件会实例化小型 Transformer。"""
import unittest
import torch
from model import MiniGPT, apply_rope


class ModelTests(unittest.TestCase):
    def test_right_padding_preserves_valid_logits_and_loss(self):
        model = MiniGPT(20, block_size=8, n_embd=8, n_head=2, n_layer=1, dropout=0).eval()
        x = torch.tensor([[1, 4, 5]])
        y = torch.tensor([[4, 5, 2]])
        padded_x = torch.tensor([[1, 4, 5, 0, 0]])
        padded_y = torch.tensor([[4, 5, 2, -100, -100]])
        short, loss = model(x, y)
        padded, padded_loss = model(padded_x, padded_y)
        torch.testing.assert_close(short, padded[:, :3])
        torch.testing.assert_close(loss, padded_loss)

    def test_switches_and_causality(self):
        for switches in ({}, {"norm_type": "rmsnorm"}, {"ffn_type": "swiglu", "ffn_hidden_dim": 12},
                         {"position_encoding": "rope"}):
            model = MiniGPT(20, block_size=8, n_embd=8, n_head=2, n_layer=1, dropout=0, **switches).eval()
            x = torch.tensor([[1, 4, 5, 6]])
            changed = x.clone()
            changed[0, -1] = 7
            a, loss = model(x, torch.tensor([[4, 5, 6, -100]]))
            b, _ = model(changed)
            self.assertEqual(a.shape, (1, 4, 20))
            self.assertTrue(torch.isfinite(loss))
            torch.testing.assert_close(a[:, :-1], b[:, :-1])
            torch.testing.assert_close(model(x[:, :2])[0], a[:, :2])
            self.assertTrue(torch.isfinite(model(x, torch.full_like(x, -100))[1]))

    def test_legacy_keys_and_rope_norm(self):
        model = MiniGPT(20, n_embd=8, n_head=2, n_layer=1)
        keys = model.state_dict()
        self.assertIn("postition_embedding.weight", keys)
        self.assertIn("blocks.0.ffn.net.0.weight", keys)
        self.assertIn("blocks.0.attn.causal_mask", keys)
        other = MiniGPT(20, n_embd=8, n_head=2, n_layer=1)
        other.load_state_dict(keys, strict=True)
        x = torch.randn(2, 2, 5, 4)
        rotated = apply_rope(x)
        torch.testing.assert_close(x.square().sum(-1), rotated.square().sum(-1))

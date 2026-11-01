"""Remote-only tensor tests. Do not run on the local development computer."""
import unittest
import torch
from model import MiniGPT
from tokenizer import CharTokenizer
from post_training.learning import (hidden_states, supervised_batch, RewardModel,
                                    ranking_loss, encode_sequence)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer=CharTokenizer.build(["春山风月夜，江水入云深。花落无人见，清光照客心。"])
        self.sample={"prompt":"春山","text":"春山风月夜，江水入云深。花落无人见，清光照客心。"}

    def test_supervision_starts_at_first_continuation_and_includes_eos(self):
        x,y=supervised_batch([self.sample],self.tokenizer,64,"cpu")
        self.assertEqual(y[0,:2].tolist(),[-100,-100])
        self.assertEqual(y[0,2].item(),self.tokenizer.stoi["风"])
        self.assertEqual(y[0,-1].item(),self.tokenizer.eos_id)
        self.assertEqual(x[0,0].item(),self.tokenizer.bos_id)

    def test_empty_prefix_learns_first_character(self):
        row=dict(self.sample,prompt="")
        _,y=supervised_batch([row],self.tokenizer,64,"cpu")
        self.assertEqual(y[0,0].item(),self.tokenizer.stoi["春"])

    def test_padding_is_ignored_and_oov_training_fails(self):
        short={"prompt":"","text":"春山"}
        _,y=supervised_batch([self.sample,short],self.tokenizer,64,"cpu")
        self.assertTrue((y[1,3:]==-100).all())
        with self.assertRaises(ValueError):
            encode_sequence({"prompt":"","text":"叽"},self.tokenizer,64)

    def test_adapter_preserves_legacy_logits_and_state_keys(self):
        for position in ("rope","learned"):
            model=MiniGPT(self.tokenizer.vocab_size,block_size=64,n_embd=12,n_head=3,n_layer=1,dropout=0.0,
                          position_encoding=position,ffn_type="swiglu").eval()
            keys=set(model.state_dict())
            ids=torch.tensor([[self.tokenizer.bos_id,self.tokenizer.stoi["春"]]])
            actual,_=model(ids)
            torch.testing.assert_close(model.lm_head(hidden_states(model,ids)),actual)
            self.assertEqual(set(model.state_dict()),keys)

    def test_reward_pools_last_nonpadding_position(self):
        model=MiniGPT(self.tokenizer.vocab_size,block_size=64,n_embd=12,n_head=3,n_layer=1,dropout=0.0).eval()
        reward=RewardModel(model,12).eval()
        a=torch.tensor([[self.tokenizer.bos_id,self.tokenizer.stoi["春"],self.tokenizer.eos_id]])
        b=torch.cat([a,torch.tensor([[self.tokenizer.pad_id,self.tokenizer.pad_id]])],dim=1)
        torch.testing.assert_close(reward(a,torch.tensor([3])),reward(b,torch.tensor([3])))

    def test_pairwise_loss_rewards_correct_order(self):
        self.assertLess(ranking_loss(torch.tensor([2.]),torch.tensor([0.])).item(),
                        ranking_loss(torch.tensor([0.]),torch.tensor([2.])).item())


if __name__ == "__main__":
    unittest.main()

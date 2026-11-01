import math
import unittest
import torch
from evaluation.metrics import generation_metrics, form_compliant, language_model_metrics, is_han


class EvaluationTests(unittest.TestCase):
    def test_han_codepoints(self):
        self.assertTrue(is_han("春"))
        self.assertTrue(is_han("\U00020000"))
        self.assertTrue(is_han("〇"))
        self.assertFalse(is_han("，"))
        self.assertFalse(is_han("a"))
        self.assertFalse(is_han("\U0002a6ff"))

    def test_text_and_target_denominators(self):
        samples = [
            {"continuation": "山山山山", "text": "山山山山", "ended_with_eos": True, "target_form": None},
            {"continuation": "", "text": "春风又绿江南岸。", "ended_with_eos": False, "target_form": "wujue"},
        ]
        metrics = generation_metrics(samples)
        self.assertAlmostEqual(metrics["distinct_1"], 1 / 4)
        self.assertAlmostEqual(metrics["repeated_bigram_ratio"], 2 / 3)
        self.assertEqual(metrics["wujue_compliance"], 0)
        self.assertIsNone(metrics["qijue_compliance"])
        self.assertIsNone(generation_metrics([])["distinct_1"])

    def test_form(self):
        self.assertTrue(form_compliant("春风吹柳绿，细雨润花红。山色连天远，江声入梦中。", "wujue"))
        self.assertFalse(form_compliant("春风吹柳绿，细雨润花红。", "wujue"))

    def test_cross_sample_phrase_not_intra_poem_repetition(self):
        result = generation_metrics([{"text": "春风"}, {"text": "春风"}])
        self.assertEqual(result["distinct_2"], 0.5)
        self.assertEqual(result["repeated_bigram_ratio"], 0)

    def test_lm_weighting_and_all_padding(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
            def forward(self, x, targets):
                return None, x[0, 0].float()
        model = FakeModel()
        model.train()
        loader = [(torch.tensor([[2]]), torch.tensor([[1]])),
                  (torch.tensor([[4, 0]]), torch.tensor([[1, 1]])),
                  (torch.tensor([[99]]), torch.tensor([[-100]]))]
        result = language_model_metrics(model, loader)
        self.assertAlmostEqual(result["loss"], 10 / 3)
        self.assertAlmostEqual(result["ppl"], math.exp(10 / 3))
        self.assertTrue(model.training)
        self.assertIsNone(language_model_metrics(model, loader[-1:])["loss"])

import unittest
import torch
from generate import apply_repetition_penalty, banned_ngram_tokens, generate, filter_logits
from tokenizer import CharTokenizer


class GenerationTests(unittest.TestCase):
    def test_penalty_and_ngram(self):
        logits = torch.tensor([-2., 2., 3.])
        penalized = apply_repetition_penalty(logits.clone(), [0, 1, 1], 2)
        torch.testing.assert_close(penalized, torch.tensor([-4., 1., 3.]))
        self.assertEqual(banned_ngram_tokens([1, 2, 1], 2), {2})
        self.assertEqual(banned_ngram_tokens([1, 2], 1), {1, 2})
        self.assertEqual(banned_ngram_tokens([], 0), set())

    def test_filters_keep_candidate(self):
        logits = filter_logits(torch.tensor([1., 2., 3.]), top_k=1, top_p=0.01)
        self.assertEqual(int(torch.isfinite(logits).sum()), 1)

    def test_hard_form_and_budget(self):
        tokenizer = CharTokenizer.build(["春山风月，。"])
        class FakeModel(torch.nn.Module):
            block_size = 64
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
            def forward(self, x):
                return torch.zeros(1, x.shape[1], tokenizer.vocab_size), None
        model = FakeModel()
        from evaluation.metrics import form_compliant
        for form, width in (("wujue", 5), ("qijue", 7)):
            result = generate(model, tokenizer, form=form, seed=42, return_details=True)
            self.assertTrue(form_compliant(result["text"], form))
            self.assertEqual(result["termination"], "constraint")
            self.assertFalse(result["ended_with_eos"])
            import re
            self.assertEqual([len(line) for line in re.split("[，。]", result["text"]) if line], [width] * 4)
        default = generate(model, tokenizer, seed=42, return_details=True)
        self.assertTrue(form_compliant(default["text"], "wujue"))
        with self.assertRaises(ValueError):
            generate(model, tokenizer, prompt="春风，", form="qijue")
        with self.assertRaises(ValueError):
            generate(model, tokenizer, form="qijue", max_new_tokens=3)


class UnknownPromptTests(unittest.TestCase):
    def model_and_tokenizer(self, block_size=64):
        tokenizer = CharTokenizer.build(['春山风月，。'])
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.block_size = block_size
                self.inputs = []
            def forward(self, x):
                self.inputs.append(x.detach().clone())
                return torch.zeros(1, x.shape[1], tokenizer.vocab_size), None
        return RecordingModel(), tokenizer

    def test_unknown_han_is_encoded_as_unk_but_preserved_in_both_forms(self):
        from evaluation.metrics import form_compliant
        for form in ('wujue', 'qijue'):
            model, tokenizer = self.model_and_tokenizer()
            result = generate(model, tokenizer, prompt='叽叽春', form=form, seed=42, return_details=True)
            self.assertTrue(result['text'].startswith('叽叽春'))
            self.assertTrue(form_compliant(result['text'], form))
            self.assertEqual(model.inputs[0][0, :4].tolist(),
                             [tokenizer.bos_id, tokenizer.unk_id, tokenizer.unk_id, tokenizer.stoi['春']])
            self.assertEqual(result['effective_prompt'], '叽叽春')
            self.assertEqual(result['text'], result['effective_prompt'] + result['continuation'])
            self.assertEqual(result['unknown_prompt_chars'], ['叽'])
            self.assertEqual(result['unknown_prompt_count'], 2)
            self.assertEqual(result['termination'], 'constraint')
            self.assertNotIn('<UNK>', result['text'])

    def test_free_mode_truncation_preserves_matching_literal_suffix(self):
        model, tokenizer = self.model_and_tokenizer(block_size=3)
        result = generate(model, tokenizer, prompt='叽春山叽', form=None,
                          max_new_tokens=0, return_details=True)
        self.assertEqual(result['effective_prompt'], '山叽')
        self.assertEqual(result['text'], '山叽')
        self.assertEqual(result['unknown_prompt_count'], 1)
        self.assertEqual(model.inputs, [])

    def test_known_prefix_still_roundtrips_without_unknowns(self):
        model, tokenizer = self.model_and_tokenizer()
        result = generate(model, tokenizer, prompt='春山', seed=42, return_details=True)
        self.assertEqual(result['text'], '春山' + result['continuation'])
        self.assertEqual(result['unknown_prompt_chars'], [])
        self.assertEqual(result['unknown_prompt_count'], 0)

    def test_unknown_non_han_is_still_invalid_for_fixed_form(self):
        model, tokenizer = self.model_and_tokenizer()
        with self.assertRaises(ValueError):
            generate(model, tokenizer, prompt='🙂', form='wujue')

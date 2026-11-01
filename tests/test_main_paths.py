"""远程回归测试：本地只编写，不执行。"""
import os
import tempfile
import unittest
from unittest.mock import patch

from config import default_config, load_config
from project_paths import PROJECT_ROOT, DEFAULT_CHECKPOINT, DEFAULT_VOCAB, project_path
from generate import build_parser, choose_form, resolve_form
from evaluate import build_parser as evaluation_parser, evaluation_inputs


class MainPathTests(unittest.TestCase):
    def test_best_configuration_and_v2_paths(self):
        cfg = default_config()
        self.assertEqual((cfg['model']['ffn_type'], cfg['model']['position_encoding'], cfg['model']['norm_type']),
                         ('swiglu', 'rope', 'layernorm'))
        self.assertEqual((cfg['training']['scheduler'], cfg['training']['learning_rate'], cfg['training']['max_steps']),
                         ('constant', 0.0003, 5000))
        for name in ('train', 'val', 'vocab'):
            self.assertEqual(project_path(cfg['data'][name]).parent, PROJECT_ROOT / 'data')
        self.assertIsNone(cfg['data']['test'])
        self.assertEqual(load_config(seed=43)['seed'], 43)

    def test_defaults_and_relative_overrides_ignore_working_directory(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                args = build_parser().parse_args([])
                self.assertEqual(args.checkpoint, DEFAULT_CHECKPOINT)
                self.assertEqual(args.vocab, DEFAULT_VOCAB)
                self.assertEqual(args.repetition_penalty, 1.1)
                explicit = build_parser().parse_args(['--checkpoint', 'checkpoints/best.pt', '--vocab', 'data/vocab.json'])
                self.assertEqual(explicit.checkpoint, DEFAULT_CHECKPOINT)
                self.assertEqual(explicit.vocab, DEFAULT_VOCAB)
                self.assertEqual(project_path('data/train.txt'), PROJECT_ROOT / 'data/train.txt')
                self.assertEqual(load_config()['data']['vocab'], 'data/vocab.json')
                self.assertEqual(project_path(load_config()['data']['vocab']), DEFAULT_VOCAB)
            finally:
                os.chdir(original)

    def test_saved_relative_data_rebinds_after_project_move(self):
        from pathlib import Path
        import project_paths
        cfg = load_config()
        self.assertEqual(cfg['data']['train'], 'data/train.txt')
        with tempfile.TemporaryDirectory() as tmp, patch.object(project_paths, 'PROJECT_ROOT', Path(tmp)):
            # 配置本身不改写；新执行位置只在文件IO前解析。
            self.assertEqual(project_paths.project_path(cfg['data']['train']), Path(tmp) / 'data/train.txt')

    def test_poem_form_selection(self):
        self.assertEqual(resolve_form(None), 'wujue')
        self.assertEqual(resolve_form('qijue'), 'qijue')
        self.assertIsNone(resolve_form('free'))
        with self.assertRaises(ValueError):
            resolve_form('unknown')
        self.assertEqual(choose_form(input_fn=lambda _: ''), 'wujue')
        self.assertEqual(choose_form(input_fn=lambda _: '2'), 'qijue')
        self.assertIsNone(choose_form(input_fn=lambda _: 'q'))
        with patch('builtins.print'):
            self.assertEqual(choose_form(input_fn=lambda _, it=iter(['invalid', '1']): next(it)), 'wujue')

    def test_evaluation_defaults_use_selected_checkpoint_and_v2(self):
        args = evaluation_parser().parse_args([])
        parent, cfg, checkpoint, paths = evaluation_inputs(args)
        self.assertIsNone(parent)
        self.assertEqual(checkpoint, DEFAULT_CHECKPOINT)
        self.assertEqual(paths['vocab'], DEFAULT_VOCAB)
        self.assertEqual(paths['val'], PROJECT_ROOT / 'data/val.txt')
        self.assertEqual(args.repetition_penalty, 1.1)
        self.assertEqual(args.evaluation_id, 'quality')


if __name__ == '__main__':
    unittest.main()

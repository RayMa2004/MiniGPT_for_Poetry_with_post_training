"""仅供用户远程运行；本地未执行。无模型依赖。"""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data_cleaning import clean_paragraphs, normalized_key, validate_rules
from prepare_data import build_dataset

ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT / "configs/data_cleaning_v2.json").read_text(encoding="utf-8"))


class CleaningTests(unittest.TestCase):
    def clean(self, *paragraphs):
        return clean_paragraphs(list(paragraphs), str, RULES)

    def test_editorial_character_is_kept(self):
        result = self.clean("[爽]节在重九，物华新雨余。")
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["text"], "爽节在重九，物华新雨余。")
        self.assertEqual(result["changes"][0]["rule"], "unwrap_editorial_character")

    def test_fullwidth_square_character_and_multiple_spans(self):
        result = self.clean("［爽］节在重九，景[龙]年物华新雨余。")
        self.assertEqual(result["text"], "爽节在重九，景龙年物华新雨余。")
        self.assertEqual(len(result["changes"]), 2)

    def test_citation_removed_but_poem_punctuation_kept(self):
        result = self.clean("家无阿堵物，门有宁馨儿。", "（见《鸡肋编》卷下）。")
        self.assertEqual(result["text"], "家无阿堵物，门有宁馨儿。")
        self.assertEqual(result["changes"][0]["removed"], "（见《鸡肋编》卷下）。")

    def test_nested_mixed_annotation_across_paragraphs(self):
        result = self.clean("首夏别京辅，杪秋滞三河。", "（〖1〗按《全唐诗》（校注[一]）。", "原卷作「别」。", "）。")
        self.assertEqual(result["text"], "首夏别京辅，杪秋滞三河。")
        self.assertEqual(len(result["changes"]), 1)

    def test_inline_explicit_variant_note(self):
        result = self.clean("无睡（张录作「眠」字）天难晓，朝夕乞时谁肯保。")
        self.assertEqual(result["text"], "无睡天难晓，朝夕乞时谁肯保。")

    def test_literary_titles_are_preserved(self):
        poem = "童子解吟《长恨曲》，胡儿能唱《琵琶篇》。"
        self.assertEqual(self.clean(poem)["text"], poem)

    def test_unknown_and_short_variants_are_quarantined(self):
        for note in ("（鹤）", "(承)", "[一]", "[注]", "〖1〗", "【春风】", "〔春风〕", "（云水苍茫）", "（按剑四顾）"):
            with self.subTest(note=note):
                result = self.clean(f"毫{note}立蛇形势未休，五天文字鬼神愁。")
                self.assertEqual(result["status"], "quarantined")
                self.assertIsNone(result["text"])

    def test_malformed_brackets_do_not_partially_clean(self):
        for text in ("[爽]节在重九（见《诗集》", "春风[秋月）江南雨。", "春风秋月）江南雨。"):
            result = self.clean(text)
            self.assertEqual(result["status"], "quarantined")
            self.assertEqual(result["changes"], [])
            self.assertIn("unbalanced_brackets", result["reasons"])

    def test_unbracketed_source_is_quarantined(self):
        result = self.clean("春风秋月江南雨，文镜秘府志所载。")
        self.assertEqual(result["status"], "quarantined")
        self.assertIn("residual_source_marker", result["reasons"])

    def test_unknown_outer_bracket_protects_whole_record(self):
        for inner in ("见《诗集》", "张录作「眠」字", "金东勋云见《诗集》"):
            result = self.clean(f"春风秋月（云水（{inner}）苍茫），江南夜雨。")
            self.assertEqual(result["status"], "quarantined")
            self.assertEqual(result["changes"], [])

    def test_invalid_empty_short_long_records(self):
        for paragraphs in (None, "正文", [1], [], [""], ["春风"], ["春" * 161]):
            result = clean_paragraphs(paragraphs, str, RULES)
            self.assertEqual(result["status"], "rejected")

    def test_whitespace_and_normalized_punctuation_key(self):
        self.assertEqual(self.clean(" 春风秋月，\n江南夜雨。 ")["text"], "春风秋月，江南夜雨。")
        self.assertEqual(normalized_key("春风，秋月。"), normalized_key("春风,秋月!"))
        self.assertNotEqual(normalized_key("春风秋月"), normalized_key("春风明月"))

    def test_rule_validation(self):
        for field, value in (("min_length", 0), ("seed", True), ("train_fraction", 1),
                             ("annotation_patterns", ["["]), ("residual_source_markers", [""])):
            rules = copy.deepcopy(RULES)
            rules[field] = value
            with self.assertRaises(ValueError):
                validate_rules(rules)


class DatasetBuildTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / "raw"
        raw.mkdir()
        records = [
            {"title": "校补", "paragraphs": ["[爽]节在重九，物华新雨余。"]},
            {"title": "重复", "paragraphs": ["爽节在重九，物华新雨余。"]},
            {"title": "同文标点", "paragraphs": ["爽节在重九。物华新雨余！"]},
            {"title": "月夜", "paragraphs": ["床前明月光，疑是地上霜。"]},
            {"title": "春晓", "paragraphs": ["春眠不觉晓，处处闻啼鸟。"]},
            {"title": "隔离", "paragraphs": ["毫（鹤）立蛇形势未休，五天文字鬼神愁。"]},
            {"title": "太短", "paragraphs": ["春风"]},
            None,
        ]
        (raw / "poet.tang.0.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return raw

    def test_deterministic_build_provenance_vocab_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.fixture(root)
            before = (raw / "poet.tang.0.json").read_bytes()
            for name in ("v2a", "v2b"):
                build_dataset(raw, root / name, RULES, shard_ids=(0,), convert=str,
                              conversion_identity={"name": "test_identity"})
            a, b = root / "v2a", root / "v2b"
            for name in ("train.txt", "val.txt", "vocab.json", "audit.jsonl"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
            train = (a / "train.txt").read_text(encoding="utf-8").splitlines()
            val = (a / "val.txt").read_text(encoding="utf-8").splitlines()
            self.assertFalse(set(map(normalized_key, train)) & set(map(normalized_key, val)))
            vocab = json.loads((a / "vocab.json").read_text(encoding="utf-8"))
            self.assertEqual(set(vocab) - {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}, set("".join(train)))
            audit = [json.loads(line) for line in (a / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(audit), 8)
            self.assertEqual(audit[1]["status"], "duplicate")
            self.assertEqual(audit[1]["duplicate_of"], audit[0]["source_id"])
            self.assertEqual(audit[0]["split"], audit[2]["split"])
            manifest = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["split"]["normalized_overlap"], 0)
            self.assertEqual(before, (raw / "poet.tang.0.json").read_bytes())
            with self.assertRaises(FileExistsError):
                build_dataset(raw, a, RULES, shard_ids=(0,), convert=str,
                              conversion_identity={"name": "test_identity"})

    def test_missing_shard_and_raw_output_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.fixture(root)
            with self.assertRaises(FileNotFoundError):
                build_dataset(raw, root / "new", RULES, convert=str,
                              conversion_identity={"name": "test_identity"})
            self.assertFalse((root / "new").exists())
            with self.assertRaises(ValueError):
                build_dataset(raw, raw / "new", RULES, shard_ids=(0,), convert=str,
                              conversion_identity={"name": "test_identity"})
            self.assertFalse((raw / "new").exists())

    def test_import_creates_no_files_and_does_not_import_opencc(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                    "import prepare_data; assert 'opencc' not in sys.modules")
            subprocess.run([sys.executable, "-c", code], cwd=tmp, check=True)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_protected_output_and_failed_build_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.fixture(root)
            for name in ("data", "checkpoints", "runs", "evaluations", "experiments", "experiments_fresh", "comparisons"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    build_dataset(raw, ROOT / name / "forbidden_v2", RULES, shard_ids=(0,), convert=str,
                                  conversion_identity={"name": "test_identity"})
            (raw / "poet.tang.0.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_dataset(raw, root / "failed", RULES, shard_ids=(0,), convert=str,
                              conversion_identity={"name": "test_identity"})
            manifest = json.loads((root / "failed/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")


if __name__ == "__main__":
    unittest.main()

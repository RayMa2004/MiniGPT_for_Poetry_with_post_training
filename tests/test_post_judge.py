"""Offline judge contract tests. Do not invoke in the static-only development session."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from post_training import judge
from post_training.rubric import DIMENSIONS, validate_result


def sample(sample_id="s1"):
    return {"task_id": "t1", "group_id": "g1", "sample_id": sample_id,
            "prompt": "春", "form": "wujue", "mode": "prefix", "split": "train",
            "text": "春山映碧流，花影落轻舟。晚照归云外，清风入小楼。",
            "continuation": "山映碧流，花影落轻舟。晚照归云外，清风入小楼。",
            "sample_index": 0, "seed": 42, "checkpoint_sha256": "a" * 64, "vocab_sha256": "b" * 64, "sampling": {"temperature": 0.8, "repetition_penalty": 1.1}}


def config():
    return {"api": {"base_url": "https://judge.example/v1", "model": "judge-model",
            "key_env": "POSTTRAIN_API_KEY", "protocol": "chat_completions",
            "response_format": "json_object", "timeout_seconds": 5,
            "max_retries": 2, "max_output_tokens": 300, "max_requests": 3}}


def response(result=None, finish="stop", refusal=None):
    result = result or {"scores": {name: 4 for name in DIMENSIONS}, "reason": "意象清楚，语意连贯。"}
    return {"choices": [{"finish_reason": finish, "message": {
        "content": json.dumps(result, ensure_ascii=False), "refusal": refusal}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}}


class JudgeTests(unittest.TestCase):
    def setUp(self):
        run_root = judge.ROOT / "post_training_runs"
        run_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=run_root)
        self.addCleanup(self.tmp.cleanup)
        self.input = Path(self.tmp.name) / "samples.jsonl"
        self.output = Path(self.tmp.name) / "labels.jsonl"
        self.write_input([sample()])
        self.calls = []

    def write_input(self, rows):
        self.input.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def transport(self, endpoint, payload, key, timeout):
        self.calls.append((endpoint, payload, key, timeout))
        return response()

    def run_judge(self, cfg=None, **kwargs):
        return judge.run_judging(cfg or config(), self.input, self.output, "score", execute=True,
                                 transport=kwargs.pop("transport", self.transport),
                                 environ={"POSTTRAIN_API_KEY": "secret-test-key"}, sleep=lambda _: None, **kwargs)

    def test_dry_run_does_not_read_key_call_transport_or_create_output(self):
        class NoSecrets(dict):
            def get(self, *args):
                raise AssertionError("dry run read environment")
        result = judge.run_judging(config(), self.input, self.output, "score", environ=NoSecrets(), transport=self.transport)
        self.assertEqual(result["pending_items"], 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.calls, [])

    def test_score_envelope_hash_usage_and_prompt_excludes_source_metadata(self):
        result = self.run_judge()
        row = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(row["item_hash"], judge.digest(sample()))
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["usage"]["total_tokens"], 50)
        self.assertEqual(result["physical_requests"], 1)
        payload = self.calls[0][1]
        self.assertEqual(payload["max_tokens"], 300)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        message = json.loads(payload["messages"][1]["content"])
        self.assertEqual(set(message["poem"]), {"text", "prefix", "form", "mode"})
        self.assertNotIn("secret-test-key", self.output.read_text(encoding="utf-8"))

    def test_resume_checks_item_hash_and_judge_fingerprint(self):
        self.run_judge()
        self.run_judge()
        self.assertEqual(len(self.calls), 1)
        changed = sample(); changed["text"] += "改"
        self.write_input([changed])
        with self.assertRaises(ValueError):
            self.run_judge()
        self.write_input([sample()])
        cfg = config(); cfg["api"]["model"] = "another-model"
        with self.assertRaises(ValueError):
            self.run_judge(cfg)

    def test_retries_count_toward_physical_cap_and_never_log_raw_errors(self):
        cfg = config(); cfg["api"]["max_requests"] = 2
        def fail(*args):
            self.calls.append(args)
            raise judge.ProviderError("http_error", status=429)
        summary = self.run_judge(cfg, transport=fail)
        self.assertEqual(summary["physical_requests"], 2)
        self.assertEqual(summary["successful_items"], 0)
        self.assertFalse(self.output.exists())
        self.assertNotIn("secret-test-key", self.output.with_suffix(".jsonl.errors.jsonl").read_text(encoding="utf-8"))

    def test_request_cap_stops_before_next_logical_item(self):
        self.write_input([sample("s1"), sample("s2"), sample("s3")])
        cfg = config(); cfg["api"]["max_requests"] = 1
        summary = self.run_judge(cfg, max_items=3)
        self.assertEqual(summary["physical_requests"], 1)
        self.assertEqual(summary["successful_items"], 1)
        self.assertEqual(summary["remaining_items"], 2)
        self.assertEqual(len(self.output.read_text(encoding="utf-8").splitlines()), 1)

    def test_timeout_then_success_uses_two_physical_requests(self):
        def transient(*args):
            self.calls.append(args)
            if len(self.calls) == 1:
                raise TimeoutError("secret-test-key must never be logged")
            return response()
        summary = self.run_judge(transport=transient)
        self.assertEqual(summary["physical_requests"], 2)
        self.assertEqual(summary["successful_items"], 1)
        log = self.output.with_suffix(".jsonl.errors.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("secret-test-key", log)
        self.assertEqual(json.loads(log)["category"], "timeout")

    def test_permanent_http_failure_is_not_retried(self):
        def fail(*args):
            self.calls.append(args)
            raise judge.ProviderError("http_error", status=401)
        self.run_judge(transport=fail)
        self.assertEqual(len(self.calls), 1)

    def test_invalid_response_is_not_labelled_or_retried(self):
        for invalid in (response(finish="length"), response(refusal="no"),
                        response({"scores": {name: True for name in DIMENSIONS}, "reason": "x"})):
            with self.subTest(invalid=invalid):
                def bad(*args):
                    self.calls.append(args)
                    return invalid
                before = len(self.calls)
                self.run_judge(transport=bad)
                self.assertEqual(len(self.calls) - before, 1)
                self.assertFalse(self.output.exists())

    def test_output_cannot_overwrite_baseline_or_input(self):
        with self.assertRaises(ValueError):
            judge.run_judging(config(), self.input, judge.ROOT / "model.py", "score")
        with self.assertRaises(ValueError):
            judge.run_judging(config(), self.input, self.input, "score")

    def test_error_sidecar_cannot_alias_input_before_key_read_or_request(self):
        collision_input = self.output.with_suffix(".jsonl.errors.jsonl")
        original = json.dumps(sample(), ensure_ascii=False) + "\n"
        collision_input.write_text(original, encoding="utf-8")
        class NoSecrets(dict):
            def get(self, *args):
                raise AssertionError("collision validation read environment")
        for execute in (False, True):
            with self.subTest(execute=execute), self.assertRaises(ValueError):
                judge.run_judging(config(), collision_input, self.output, "score",
                                  execute=execute, environ=NoSecrets(), transport=self.transport)
        self.assertEqual(collision_input.read_text(encoding="utf-8"), original)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.calls, [])

    def test_endpoint_rejects_redirects_credentials_query_and_unsafe_http(self):
        for url in ("http://judge.example/v1", "https://key@judge.example/v1", "https://judge.example/v1?key=x", "https://judge.example/#key"):
            cfg = config(); cfg["api"]["base_url"] = url
            with self.assertRaises(ValueError):
                self.run_judge(cfg)
        with self.assertRaises(judge.ProviderError):
            judge.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://other.example")
        self.assertEqual(self.calls, [])

    def test_pair_rejects_cross_task_and_accepts_tie(self):
        pair = {"pair_id": "p1", "task_id": "t1", "split": "train", "a": sample(), "b": sample("s2")}
        self.write_input([pair])
        def tie(*args):
            return response({"winner": "tie", "reason": "整体质量接近。"})
        judge.run_judging(config(), self.input, self.output, "pairs", execute=True,
                         transport=tie, environ={"POSTTRAIN_API_KEY": "secret-test-key"})
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8"))["result"]["winner"], "tie")
        pair["b"]["task_id"] = "other"
        self.write_input([pair])
        with self.assertRaises(ValueError):
            judge.run_judging(config(), self.input, self.output, "pairs")

    def test_explicit_response_modes_and_generation_parameters(self):
        for mode in ("json_schema", "json_object", "plain"):
            cfg = config(); cfg["api"]["response_format"] = mode
            normalized = judge.normalize_api(cfg)
            payload = judge.build_payload(normalized, "score", sample())
            self.assertEqual("response_format" in payload, mode != "plain")
            if mode == "json_schema":
                self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        cfg = config(); cfg["api"]["response_format"] = "auto"
        with self.assertRaises(ValueError):
            judge.normalize_api(cfg)

    def test_explicit_null_temperature_omits_parameter_for_reasoning_providers(self):
        cfg = config()
        default_payload = judge.build_payload(judge.normalize_api(cfg), "score", sample())
        self.assertEqual(default_payload["temperature"], 0)
        cfg["api"]["temperature"] = None
        api = judge.normalize_api(cfg)
        payload = judge.build_payload(api, "score", sample())
        self.assertNotIn("temperature", payload)
        self.assertNotEqual(judge.judge_fingerprint(api, "score"),
                            judge.judge_fingerprint(judge.normalize_api(config()), "score"))

    def test_result_validation_rejects_extra_keys_out_of_range_and_long_reason(self):
        good = {"scores": {name: 4 for name in DIMENSIONS}, "reason": "ok"}
        for value in (0, 6, 4.0, True, float("nan")):
            bad = copy.deepcopy(good); bad["scores"]["imagery"] = value
            with self.assertRaises(ValueError):
                validate_result("score", bad)
        bad = copy.deepcopy(good); bad["extra"] = 1
        with self.assertRaises(ValueError):
            validate_result("score", bad)
        with self.assertRaises(ValueError):
            validate_result("pairs", {"winner": "A", "reason": "x" * 801})


if __name__ == "__main__":
    unittest.main()

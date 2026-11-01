"""Explicit, resumable third-party Chat Completions judging using stdlib only.

Importing this module never reads a secret or sends a request. Each execute run's
max_requests counts physical attempts, including retries. Existing successful
labels are reused only after both content and the judge fingerprint match.
"""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .rubric import RUBRIC_VERSION, SYSTEM_PROMPT, result_schema, validate_result
from .core import ROOT, artifact_path, digest, validate_sample

PROMPT_VERSION = "blind-chat-poetry-v1"


def _json_loads(text):
    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError("non-finite JSON number")
    return json.loads(text, object_pairs_hook=reject_duplicate, parse_constant=reject_constant)


def _read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [_json_loads(line) for line in handle if line.strip()]


def _append(path, row):
    path = artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _positive_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("invalid " + name)
    return value


def normalize_api(config):
    raw = config.get("api")
    if not isinstance(raw, dict):
        raise ValueError("config.api must be an object")
    api = dict(raw)
    if api.get("protocol", "chat_completions") != "chat_completions":
        raise ValueError("only chat_completions protocol is supported")
    mode = api.get("response_format", "json_object")
    if mode not in ("json_schema", "json_object", "plain"):
        raise ValueError("response_format must be explicit json_schema, json_object or plain")
    if api.get("key_env", "POSTTRAIN_API_KEY") != "POSTTRAIN_API_KEY":
        raise ValueError("the only supported secret environment variable is POSTTRAIN_API_KEY")
    base = api.get("base_url", "")
    model = api.get("model", "")
    if not isinstance(base, str) or not isinstance(model, str):
        raise ValueError("base_url and model must be strings")
    endpoint = ""
    if base:
        parsed = urlsplit(base)
        try:
            parsed.port
        except ValueError:
            raise ValueError("invalid endpoint port") from None
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("endpoint must not contain credentials, query or fragment")
        if not parsed.hostname or any(char.isspace() for char in base) or "\\" in base:
            raise ValueError("invalid endpoint")
        local_http = (parsed.scheme == "http" and api.get("allow_localhost_http") is True
                      and parsed.hostname.lower() in ("localhost", "127.0.0.1", "::1"))
        if parsed.scheme != "https" and not local_http:
            raise ValueError("HTTPS required; localhost HTTP needs allow_localhost_http=true")
        path = parsed.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        endpoint = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    max_requests = api.get("max_requests", 200)
    retries = api.get("max_retries", 2)
    tokens = api.get("max_output_tokens", 1200)
    for name, value, minimum in (("max_requests", max_requests, 1), ("max_retries", retries, 0), ("max_output_tokens", tokens, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError("invalid " + name)
    timeout = _positive_number(api.get("timeout_seconds", 90), "timeout_seconds")
    token_parameter = api.get("token_parameter", "max_tokens")
    if token_parameter not in ("max_tokens", "max_completion_tokens"):
        raise ValueError("invalid token_parameter")
    generation = {token_parameter: tokens}
    if "thinking" in api:
        thinking = api["thinking"]
        if not isinstance(thinking, dict) or set(thinking) != {"type"} or thinking["type"] not in ("enabled", "disabled"):
            raise ValueError("thinking must be an object with type enabled/disabled")
        generation["thinking"] = dict(thinking)
    batch_size = api.get("batch_size", 1)
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be an integer from 1 to 32")
    # Explicit null permits providers that accept only their temperature default.
    temperature = api.get("temperature", 0)
    if temperature is not None:
        generation["temperature"] = temperature
    for name in ("top_p", "seed"):
        if name in api:
            generation[name] = api[name]
    if temperature is not None and (type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError("invalid temperature")
    if "top_p" in generation:
        _positive_number(generation["top_p"], "top_p")
        if generation["top_p"] > 1:
            raise ValueError("invalid top_p")
    if "seed" in generation and type(generation["seed"]) is not int:
        raise ValueError("seed must be an integer")
    return {"endpoint": endpoint, "model": model, "protocol": "chat_completions", "response_format": mode,
            "timeout_seconds": timeout, "max_retries": retries, "max_requests": max_requests,
            "generation": generation, "batch_size": batch_size, "key_env": "POSTTRAIN_API_KEY"}


def _sample(sample):
    if not isinstance(sample, dict):
        raise ValueError("sample must be an object")
    validate_sample(sample)
    for field in ("task_id", "sample_id", "text"):
        if not isinstance(sample.get(field), str) or not sample[field].strip():
            raise ValueError("invalid sample " + field)
    if not isinstance(sample.get("prompt"), str):
        raise ValueError("invalid literal prefix")
    if sample.get("form") not in ("wujue", "qijue") or sample.get("mode") not in ("prefix", "free"):
        raise ValueError("invalid form or mode")
    if sample.get("split") not in ("train", "val", "test"):
        raise ValueError("invalid split")
    if (sample["mode"] == "free") != (sample["prompt"] == ""):
        raise ValueError("mode and literal prefix disagree")
    return {"text": sample["text"], "prefix": sample["prompt"], "form": sample["form"], "mode": sample["mode"]}


def _item_id(kind, item):
    if kind == "score":
        _sample(item)
        return item["sample_id"]
    if kind != "pairs" or not isinstance(item, dict):
        raise ValueError("invalid judging kind or pair")
    if not isinstance(item.get("pair_id"), str) or not item["pair_id"].strip():
        raise ValueError("invalid pair id")
    a, b = item.get("a"), item.get("b")
    _sample(a); _sample(b)
    if a["sample_id"] == b["sample_id"]:
        raise ValueError("pair must contain distinct samples")
    for field in ("task_id", "group_id", "form", "mode", "prompt", "split"):
        if a[field] != b[field]:
            raise ValueError("pair must share task, form, mode, prefix and split")
    if item.get("task_id") != a["task_id"] or item.get("split") != a["split"]:
        raise ValueError("pair identity mismatch")
    return item["pair_id"]


def build_payload(api, kind, item):
    _item_id(kind, item)
    content = {"task": kind, "poem": _sample(item)} if kind == "score" else {
        "task": "pairs", "A": _sample(item["a"]), "B": _sample(item["b"])}
    payload = {"model": api["model"], **api["generation"], "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(content, ensure_ascii=False)}]}
    if api["response_format"] == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif api["response_format"] == "json_schema":
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "poetry_" + kind, "strict": True, "schema": result_schema(kind)}}
    return payload


def judge_fingerprint(api, kind):
    identity = {"rubric_version": RUBRIC_VERSION, "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT,
                   "schema": result_schema(kind), "kind": kind, "endpoint": api["endpoint"],
                   "model": api["model"], "protocol": api["protocol"],
                   "response_format": api["response_format"], "generation": api["generation"]}
    if api["batch_size"] > 1:
        identity.update(batch_version="independent-items-v1", batch_size=api["batch_size"],
                        batch_instruction=BATCH_INSTRUCTION)
    return digest(identity)


BATCH_INSTRUCTION = """本次输入包含多个独立评审条目，返回JSON对象：
{"items":[{"item_id":"输入中的ID","result":{"scores":{"fluency":3,"coherence":3,"imagery":3,"poeticness":3},"reason":"简短理由"}}]}。
每个输入ID必须恰好返回一次，不遗漏、不添加、不改变ID。评分使用上面的固定绝对尺度，
不能按本批次相对排名分配分数，不要求存在高分作品。每条理由尽量控制在60个汉字内。
若task为pairs，result改为{"winner":"A或B或tie或both_bad","reason":"简短理由"}。
逐项独立判断，不混合不同条目的诗句，不改写诗作。"""


def build_batch_payload(api, kind, entries):
    """Opaque item IDs bind independently evaluated results to local artifacts."""
    contents = []
    for item_id, item, _ in entries:
        single = build_payload(api, kind, item)
        contents.append(dict(json.loads(single["messages"][1]["content"]), item_id=item_id))
    payload = {"model": api["model"], **api["generation"], "messages": [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + BATCH_INSTRUCTION},
        {"role": "user", "content": json.dumps({"items": contents}, ensure_ascii=False)}]}
    if api["response_format"] == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif api["response_format"] == "json_schema":
        schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": {"item_id": {"type": "string"}, "result": result_schema(kind)},
            "required": ["item_id", "result"], "additionalProperties": False}}},
            "required": ["items"], "additionalProperties": False}
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "poetry_batch_" + kind, "strict": True, "schema": schema}}
    return payload


def _parse_batch_response(kind, response, key, expected):
    value, usage = _response_content(response, key)
    try:
        if not isinstance(value, dict) or set(value) != {"items"} or not isinstance(value["items"], list):
            raise ValueError("invalid batch envelope")
        results = {}
        for row in value["items"]:
            if not isinstance(row, dict) or set(row) != {"item_id", "result"}:
                raise ValueError("invalid batch item")
            identity = row["item_id"]
            if not isinstance(identity, str) or identity not in expected or identity in results:
                raise ValueError("unknown/duplicate item ID")
            results[identity] = validate_result(kind, row["result"])
        if set(results) != set(expected):
            raise ValueError("missing item ID")
        return results, usage
    except (ValueError, TypeError, KeyError):
        raise ProviderError("invalid_response") from None


class ProviderError(Exception):
    """Only safe fixed categories and numeric HTTP status may reach logs."""
    CATEGORIES = {"http_error", "timeout", "network_error", "redirect_refused", "invalid_response", "request_cap"}

    def __init__(self, category, status=None):
        self.category = category if category in self.CATEGORIES else "network_error"
        self.status = status if type(status) is int and 100 <= status <= 599 else None
        super().__init__(self.category)

    @property
    def retryable(self):
        return self.category == "timeout" or (self.category == "http_error" and
               (self.status == 429 or self.status is not None and 500 <= self.status <= 599))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("redirect_refused", status=code)


def http_transport(endpoint, payload, key, timeout):
    """One physical request; refuse redirects and never retain provider errors."""
    request = Request(endpoint, data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Accept": "application/json", "Authorization": "Bearer " + key}, method="POST")
    try:
        with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024:
                raise ProviderError("invalid_response")
            return _json_loads(body.decode("utf-8"))
    except HTTPError as error:
        status = error.code
        error.close()
        raise ProviderError("http_error", status=status) from None
    except (TimeoutError, socket.timeout):
        raise ProviderError("timeout") from None
    except URLError as error:
        category = "timeout" if isinstance(error.reason, (TimeoutError, socket.timeout)) else "network_error"
        raise ProviderError(category) from None
    except (UnicodeError, ValueError):
        raise ProviderError("invalid_response") from None


def _response_content(response, key):
    try:
        if not isinstance(response, dict) or not isinstance(response.get("choices"), list) or len(response["choices"]) != 1:
            raise ValueError("invalid choices")
        choice = response["choices"][0]
        message = choice["message"]
        if choice.get("finish_reason") != "stop" or message.get("refusal") or message.get("tool_calls") or message.get("function_call"):
            raise ValueError("unfinished or refused response")
        content = message["content"]
        if not isinstance(content, str) or len(content) > 65536 or key in content:
            raise ValueError("invalid content")
        result = _json_loads(content)
        usage = response.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("invalid usage")
        safe_usage = {name: usage[name] for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                      if name in usage and type(usage[name]) is int and usage[name] >= 0}
        return result, safe_usage
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        raise ProviderError("invalid_response") from None


def _parse_response(kind, response, key):
    result, usage = _response_content(response, key)
    try:
        return validate_result(kind, result), usage
    except (ValueError, TypeError, KeyError):
        raise ProviderError("invalid_response") from None


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def run_judging(config, input_path, output_path, kind, *, execute=False, max_items=None,
                 transport=None, environ=None, sleep=time.sleep):
    api = normalize_api(config)
    if max_items is None:
        max_items = api["max_requests"] * api["batch_size"]
    if type(max_items) is not int or max_items < 1:
        raise ValueError("max_items must be positive")
    input_path, output_path = Path(input_path).resolve(), artifact_path(output_path)
    error_path = artifact_path(output_path.with_suffix(output_path.suffix + ".errors.jsonl"))
    def aliases(first, second):
        return first == second or (first.exists() and second.exists() and first.samefile(second))
    if any(aliases(input_path, target) for target in (output_path, error_path)):
        raise ValueError("input must differ from label and error output paths")
    if aliases(output_path, error_path):
        raise ValueError("label and error output paths must differ")
    items = _read_rows(input_path)
    indexed = {}
    for item in items:
        item_id = _item_id(kind, item)
        if item_id in indexed:
            raise ValueError("duplicate input item id")
        indexed[item_id] = (item, digest(item))
    fingerprint = judge_fingerprint(api, kind)
    cached = {}
    if output_path.exists():
        for row in _read_rows(output_path):
            if not isinstance(row, dict) or row.get("status") != "ok" or row.get("judge_id") != fingerprint:
                raise ValueError("cache judge fingerprint/status mismatch")
            item_id = row.get("item_id")
            if not isinstance(item_id, str) or item_id in cached:
                raise ValueError("invalid or duplicate cached item id")
            validate_result(kind, row.get("result"))
            if item_id not in indexed or row.get("item_hash") != indexed[item_id][1]:
                raise ValueError("cache input content mismatch")
            cached[item_id] = row
    pending = [(item_id, item, item_hash) for item_id, (item, item_hash) in indexed.items() if item_id not in cached][:max_items]
    summary = {"execute": execute, "kind": kind, "input_items": len(items), "cached_items": len(cached),
               "pending_items": len(pending), "max_items": max_items, "max_requests": api["max_requests"],
               "judge_id": fingerprint, "physical_requests": 0, "successful_items": 0, "failed_items": 0,
               "remaining_items": len(items) - len(cached), "batch_size": api["batch_size"],
               "estimated_requests": 0}
    grouped = defaultdict(list)
    for entry in pending:
        item = entry[1]
        grouped[(item["task_id"], item["split"])].append(entry)
    batches = deque(batch[start:start + api["batch_size"]]
                    for batch in grouped.values() for start in range(0, len(batch), api["batch_size"]))
    summary["estimated_requests"] = len(batches)
    if not execute or not pending:
        return summary
    if not api["endpoint"] or not api["model"].strip():
        raise ValueError("execute requires configured API base_url and model")
    environ = os.environ if environ is None else environ
    key = environ.get(api["key_env"])
    if not isinstance(key, str) or not key.strip() or any(char.isspace() for char in key):
        raise ValueError("execute requires a nonempty POSTTRAIN_API_KEY without whitespace")
    # Never persist a credential if it was accidentally pasted into model/config.
    if key in json.dumps(api, ensure_ascii=False) or key in json.dumps(items, ensure_ascii=False):
        raise ValueError("credential must appear only in the secret environment variable")
    transport = http_transport if transport is None else transport
    while batches and summary["physical_requests"] < api["max_requests"]:
        entries = batches.popleft()
        batched = api["batch_size"] > 1
        payload = build_batch_payload(api, kind, entries) if batched else build_payload(api, kind, entries[0][1])
        for attempt in range(api["max_retries"] + 1):
            if summary["physical_requests"] >= api["max_requests"]:
                break
            summary["physical_requests"] += 1
            try:
                response = transport(api["endpoint"], payload, key, api["timeout_seconds"])
                if batched:
                    results, usage = _parse_batch_response(kind, response, key, {e[0] for e in entries})
                else:
                    result, usage = _parse_response(kind, response, key)
                    results = {entries[0][0]: result}
            except Exception as error:
                safe = error if isinstance(error, ProviderError) else ProviderError(
                    "timeout" if isinstance(error, (TimeoutError, socket.timeout)) else "network_error")
                retry = safe.retryable and attempt < api["max_retries"] and summary["physical_requests"] < api["max_requests"]
                split_batch = (batched and len(entries) > 1 and
                               (safe.category == "invalid_response" or safe.retryable) and not retry)
                for item_id, item, item_hash in entries:
                    _append(error_path, {"item_id": item_id, "item_hash": item_hash, "judge_id": fingerprint,
                        "timestamp": _timestamp(), "attempt": attempt + 1,
                        "category": safe.category, "http_status": safe.status,
                        "will_retry": retry or split_batch})
                if retry:
                    sleep(min(2 ** attempt, 8))
                    continue
                if split_batch:
                    midpoint = len(entries) // 2
                    batches.appendleft(entries[midpoint:])
                    batches.appendleft(entries[:midpoint])
                else:
                    summary["failed_items"] += len(entries)
                break
            request_id = digest({"items": [e[0] for e in entries], "timestamp": _timestamp(),
                                 "judge_id": fingerprint, "request": summary["physical_requests"]})[:24]
            for position, (item_id, item, item_hash) in enumerate(entries):
                # Batch usage is stored once, not multiplied by the number of poems.
                row = {"item_id": item_id, "item_hash": item_hash, "judge_id": fingerprint,
                    "status": "ok", "result": results[item_id], "usage": usage if position == 0 else {},
                    "timestamp": _timestamp(), "provider_model": api["model"],
                    "rubric_version": RUBRIC_VERSION, "protocol": api["protocol"],
                    "response_format": api["response_format"]}
                if batched:
                    row.update(request_id=request_id, batch_items=len(entries), usage_scope="request")
                _append(output_path, row)
                summary["successful_items"] += 1
                summary["remaining_items"] -= 1
            print(f"Judged {len(items) - summary['remaining_items']}/{len(items)}; "
                  f"requests={summary['physical_requests']}/{api['max_requests']}", flush=True)
            break
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="AI poetry judge; dry run unless --execute is supplied")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "pairs"):
        command = subs.add_parser(name)
        command.add_argument("--input", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--config", default=str(ROOT / "configs" / "post_training_v3.json"))
        command.add_argument("--max-items", type=int)
        command.add_argument("--batch-size", type=int, help="Items per request within one task, 1-32")
        command.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else ROOT / path
    try:
        with resolve(args.config).open(encoding="utf-8-sig") as handle:
            config = _json_loads(handle.read())
        if args.batch_size is not None:
            config["api"]["batch_size"] = args.batch_size
        summary = run_judging(config, resolve(args.input), resolve(args.output), args.command,
                              execute=args.execute, max_items=args.max_items)
    except (OSError, ValueError) as error:
        # Do not echo arbitrary paths/config values or provider data to stderr.
        parser.exit(2, "judge: invalid local configuration, input, or resume cache (" + type(error).__name__ + ")\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_items"] == 0 and (not args.execute or summary["remaining_items"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())

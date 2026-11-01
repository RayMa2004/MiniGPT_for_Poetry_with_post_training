"""Stdlib contracts and immutable identities; no training or HTTP at import."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from project_paths import PROJECT_ROOT
from experiment import atomic_json
from evaluation.metrics import form_compliant, is_han

ROOT = PROJECT_ROOT
DIMENSIONS = ("fluency", "coherence", "imagery", "poeticness")
SPLITS = ("train", "val", "test")
FORMS = ("wujue", "qijue")


def resolve(path):
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{number}: JSON object required")
                records.append(item)
    return records


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n")


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def artifact_path(value):
    path = resolve(value)
    allowed = [ROOT / name for name in ("post_training_data", "post_training_runs")]
    if not any(path.is_relative_to(base.resolve()) and path != base.resolve() for base in allowed):
        raise ValueError("后训练输出必须位于项目 post_training_data/ 或 post_training_runs/ 子目录")
    return path


def new_output(value):
    path = artifact_path(value)
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_config(path=None):
    config = read_json(resolve(path or "configs/post_training_v3.json"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported post-training config schema")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Invalid seed")
    if set(config.get("paths", {})) != {"checkpoint", "vocab", "output_root"}:
        raise ValueError("Missing checkpoint/vocab/output_root")
    for group in ("sft", "reward", "ppo", "selection"):
        if not isinstance(config.get(group), dict):
            raise ValueError(f"Missing config section: {group}")
        for name, value in config[group].items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"Nonfinite config {group}.{name}")
    return config


def validate_task(task):
    required = {"task_id", "group_id", "prompt", "form", "mode", "split"}
    if required - task.keys() or task["form"] not in FORMS or task["split"] not in SPLITS:
        raise ValueError("Invalid task identity/form/split")
    if not isinstance(task["prompt"], str) or len(task["prompt"]) > 4 or any(not is_han(ch) for ch in task["prompt"]):
        raise ValueError("训练开头必须为0–4个汉字；不接受长指令")
    if task["mode"] != ("prefix" if task["prompt"] else "free"):
        raise ValueError("mode disagrees with prompt")
    if any(not isinstance(task[key], str) or not task[key] for key in ("task_id", "group_id")):
        raise ValueError("Task/group ID required")
    return task


def validate_sample(sample):
    validate_task(sample)
    required = {"sample_id", "text", "continuation", "sample_index", "seed", "checkpoint_sha256", "vocab_sha256", "sampling"}
    if required - sample.keys():
        raise ValueError("Missing sample provenance")
    if type(sample["sample_index"]) is not int or sample["sample_index"] < 0 or type(sample["seed"]) is not int or not 0 <= sample["seed"] < 2**32:
        raise ValueError("Invalid sample index/seed")
    if not isinstance(sample["sampling"], dict) or not sample["sampling"]:
        raise ValueError("Missing sampling settings")
    if not isinstance(sample["sample_id"], str) or not sample["sample_id"]:
        raise ValueError("Invalid sample ID")
    if not isinstance(sample["text"], str) or not sample["text"].startswith(sample["prompt"]):
        raise ValueError("Sample changed prefix")
    if sample["continuation"] != sample["text"][len(sample["prompt"]):]:
        raise ValueError("Continuation does not match literal prefix")
    if any(ch.isspace() for ch in sample["text"]) or not form_compliant(sample["text"], sample["form"]):
        raise ValueError("Sample fails poem format")
    for name in ("checkpoint_sha256", "vocab_sha256"):
        if not isinstance(sample[name], str) or len(sample[name]) != 64 or any(ch not in "0123456789abcdef" for ch in sample[name]):
            raise ValueError("Invalid SHA256 provenance")
    return sample


def unique_index(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError(f"Duplicate {key}: {identity}")
        result[identity] = row
    return result


def validate_group_splits(rows):
    groups, prefixes = {}, {}
    for row in rows:
        validate_task(row)
        for table, key in ((groups, row["group_id"]), (prefixes, row["prompt"] if row["prompt"] else None)):
            if key is not None:
                if key in table and table[key] != row["split"]:
                    raise ValueError("同一开头或任务组跨训练/验证/测试集合")
                table[key] = row["split"]


def source_snapshot():
    paths = sorted((ROOT/"post_training").glob("*.py")) + [ROOT/"model.py", ROOT/"checkpoint.py", ROOT/"tokenizer.py", ROOT/"generate.py"]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def read_complete_samples(path):
    path=Path(path)
    manifest=read_json(path.parent/"manifest.json")
    rows=read_jsonl(path)
    if manifest.get("status") != "completed" or manifest.get("samples_sha256") != file_hash(path):
        raise ValueError("候选采样未完整结束或内容哈希已变化")
    if manifest.get("sample_count") != len(rows) or manifest.get("expected_count") != len(rows):
        raise ValueError("候选采样数量不完整")
    return rows

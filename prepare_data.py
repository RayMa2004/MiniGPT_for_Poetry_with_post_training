"""显式构建独立的唐诗数据版本。只读本地 raw，无下载、无导入时副作用。

完整构建及测试仅供用户远程执行。本地仅允许阅读源码与静态检查。
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from data_cleaning import clean_paragraphs, normalized_key, validate_rules
from project_paths import project_path

SHARD_IDS = tuple(range(0, 58000, 1000))
ROOT = Path(__file__).resolve().parent


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def _json(path, payload):
    # 仅写本次新建的独立目录；manifest 以临时文件替换，避免半份完成状态。
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _output_path(raw_dir, output_dir):
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    protected = [ROOT / name for name in ("data", "checkpoints", "runs", "evaluations", "experiments", "experiments_fresh", "comparisons")]
    protected = [path.resolve() for path in protected] + [raw_dir]
    if any(output_dir == path or path in output_dir.parents for path in protected):
        raise ValueError("输出不得位于原始数据、权重、实验或历史报告目录中")
    if output_dir == ROOT or output_dir in raw_dir.parents or output_dir in ROOT.parents:
        raise ValueError("输出必须是新的独立数据目录")
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录: {output_dir}")
    return output_dir


def _opencc_converter():
    from opencc import OpenCC
    converter = OpenCC("t2s")
    return converter.convert, {"name": "OpenCC", "config": "t2s",
                               "package": "opencc-python-reimplemented",
                               "version": importlib.metadata.version("opencc-python-reimplemented")}


def build_dataset(raw_dir, output_dir, rules, *, shard_ids=SHARD_IDS, convert=None,
                  conversion_identity=None, rules_path=None):
    """仅显式调用时构建。convert 注入用于远程小夹具测试，必须声明转换身份。"""
    validate_rules(rules)
    raw_dir = Path(raw_dir).resolve()
    output_dir = _output_path(raw_dir, output_dir)
    if not shard_ids or len(set(shard_ids)) != len(shard_ids) or any(type(i) is not int or i < 0 for i in shard_ids):
        raise ValueError("shard_ids 必须为非负整数且不能重复")
    paths = [raw_dir / f"poet.tang.{i}.json" for i in sorted(shard_ids)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"缺少本地 raw 分片（不会自动下载）: {path}")
    if convert is None:
        convert, conversion_identity = _opencc_converter()
    elif not conversion_identity or not callable(convert):
        raise ValueError("自定义转换必须提供 callable 和 conversion_identity")
    from tokenizer import CharTokenizer

    # 原子占用全新目录。异常不清理，保留失败状态和证据；重试需新版本目录。
    output_dir.mkdir(parents=True, exist_ok=False)
    rule_bytes = json.dumps(rules, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {"schema_version": 1, "status": "building",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rule_version": rules["rule_version"], "rules": rules,
                "rules_canonical_sha256": hashlib.sha256(rule_bytes).hexdigest(),
                "rules_file": fingerprint(rules_path) if rules_path else None,
                "raw_dir": str(raw_dir), "shard_ids": sorted(shard_ids), "raw_files": {},
                "source_files": {name: fingerprint(ROOT / name) for name in
                                 ("prepare_data.py", "data_cleaning.py", "tokenizer.py", "project_paths.py")},
                "environment": {"python": platform.python_version(), "unicode": unicodedata.unidata_version,
                                "conversion": conversion_identity},
                "test_split": None, "vocabulary_scope": "train_only"}
    _json(output_dir / "manifest.json", manifest)
    try:
        audit, unique, owners = [], [], {}
        for path in paths:
            before = fingerprint(path)
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError(f"raw 分片不是列表: {path.name}")
            if fingerprint(path) != before:
                raise RuntimeError(f"读取时 raw 分片发生变化: {path.name}")
            manifest["raw_files"][path.name] = before
            for index, item in enumerate(records):
                row = {"source_id": f"{path.name}:{index}", "shard": path.name,
                       "record_index": index, "split": None}
                if not isinstance(item, dict):
                    row.update(status="rejected", reasons=["invalid_record"], original_record=item,
                               text=None, normalized_text=None, changes=[], brackets=[])
                else:
                    row.update(title=item.get("title"), author=item.get("author"),
                               original_paragraphs=item.get("paragraphs"))
                    row.update(clean_paragraphs(item.get("paragraphs"), convert, rules))
                if row["status"] == "accepted":
                    text = row["text"]
                    if text in owners:
                        row.update(status="duplicate", reasons=["exact_duplicate_after_cleaning"],
                                   duplicate_of=owners[text]["source_id"])
                    else:
                        owners[text] = row
                        unique.append(row)
                audit.append(row)

        groups = {}
        for row in unique:
            groups.setdefault(normalized_key(row["text"]), []).append(row)
        keys = sorted(groups)
        random.Random(rules["seed"]).shuffle(keys)
        split_index = int(len(keys) * rules["train_fraction"])
        if not 0 < split_index < len(keys):
            raise ValueError("清洗后分组不足以产生非空 train/val")
        splits = {"train": [], "val": []}
        for index, key in enumerate(keys):
            split = "train" if index < split_index else "val"
            for row in groups[key]:
                row["split"] = split
                splits[split].append(row["text"])
        for row in audit:
            if row["status"] == "duplicate":
                row["split"] = owners[row["text"]]["split"]
        exact_overlap = len(set(splits["train"]) & set(splits["val"]))
        normalized_overlap = len(set(map(normalized_key, splits["train"])) & set(map(normalized_key, splits["val"])))
        if exact_overlap or normalized_overlap:
            raise RuntimeError("跨集合重复检查失败")
        for split, texts in splits.items():
            with (output_dir / f"{split}.txt").open("x", encoding="utf-8", newline="\n") as handle:
                handle.writelines(text + "\n" for text in texts)
        # 每个版本独立训练词表；验证集不参与词表构建。
        tokenizer = CharTokenizer.build(splits["train"])
        tokenizer.save(output_dir / "vocab.json")
        with (output_dir / "audit.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in audit:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        statuses = Counter(row["status"] for row in audit)
        reasons = Counter(reason for row in audit for reason in row["reasons"])
        operations = Counter(change["rule"] for row in audit for change in row["changes"])
        accepted_operations = Counter(change["rule"] for row in unique for change in row["changes"])
        val_chars = Counter("".join(splits["val"]))
        unknown = {ch: count for ch, count in sorted(val_chars.items()) if ch not in tokenizer.stoi}
        manifest.update(status="completed", counts={"raw_records": len(audit), **dict(statuses)},
                        reasons=dict(reasons), candidate_operations=dict(operations), accepted_operations=dict(accepted_operations),
                        split={"seed": rules["seed"], "train_fraction_groups": rules["train_fraction"],
                               "group_count": len(groups), "multi_text_group_count": sum(len(g) > 1 for g in groups.values()),
                               "key": "remove_whitespace_and_Unicode_P_punctuation",
                               "exact_overlap": exact_overlap, "normalized_overlap": normalized_overlap,
                               "near_duplicate_check": "not_performed",
                               **{name: {"poems": len(texts), "characters": sum(map(len, texts))}
                                  for name, texts in splits.items()}},
                        vocabulary={"size": tokenizer.vocab_size, "scope": "train_only",
                                    "val_unknown_characters": unknown,
                                    "val_unknown_count": sum(unknown.values()),
                                    "val_unknown_rate": sum(unknown.values()) / sum(val_chars.values())},
                        output_files={name: fingerprint(output_dir / name) for name in
                                      ("train.txt", "val.txt", "vocab.json", "audit.jsonl")})
        _json(output_dir / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _json(output_dir / "manifest.json", manifest)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description="从现有 raw 显式构建新版本；绝不覆盖原 data，无自动下载")
    parser.add_argument("--raw-dir", type=project_path, required=True, help="外部原始分片目录；本项目已删除旧raw，不会自动下载")
    parser.add_argument("--output-dir", type=project_path, required=True, help="必须是新版本目录，例如 data_v3；当前data受保护")
    parser.add_argument("--rules", type=project_path, default=ROOT / "configs/data_cleaning_v2.json")
    args = parser.parse_args()
    rules_path = Path(args.rules)
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    manifest = build_dataset(args.raw_dir, args.output_dir, rules, rules_path=rules_path)
    print(json.dumps({"output": str(Path(args.output_dir).resolve()), "status": manifest["status"],
                      "counts": manifest["counts"], "split": manifest["split"],
                      "vocabulary": manifest["vocabulary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Torch-free prepared artifact validation and actionable readiness reports."""
from pathlib import Path
from .core import resolve, read_json, read_jsonl, file_hash, validate_sample, validate_group_splits

EXPECTED_TYPES = {f"{mode}/{form}" for mode in ("prefix", "free") for form in ("wujue", "qijue")}


def readiness(splits):
    missing = {}
    counts = {}
    for split in ("train", "val"):
        rows = splits.get(split, [])
        observed = {f"{row['mode']}/{row['form']}" for row in rows}
        missing[split] = sorted(EXPECTED_TYPES - observed)
        counts[split] = len(rows)
    return {"ready": not any(missing.values()), "counts": counts, "missing": missing}


def require_coverage(rows, name):
    observed = {f"{row['mode']}/{row['form']}" for row in rows}
    missing = sorted(EXPECTED_TYPES - observed)
    if missing:
        raise ValueError(f"{name}: {len(rows)} usable rows; missing {', '.join(missing)}. "
                         "Increase/calibrate qualified candidates; do not copy train rows into val.")


def prepared_manifest(directory):
    directory = resolve(directory)
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Prepared manifest missing: {path}. The data prepare step has not completed. "
                         "Run python -m post_training data prepare with --samples, --scores and "
                         f"--output {directory}, or use the pipeline command in docs/POST_TRAINING_V3.md. "
                         "Creating an empty directory does not prepare data.")
    return read_json(path)


def load_prepared(directory, section, *, require_ready=True):
    directory=resolve(directory)
    manifest=prepared_manifest(directory)
    if section not in ("sft", "preferences"):
        raise ValueError("section must be sft or preferences")
    result={}
    for split in ("train","val"):
        name=f"{section}/{split}.jsonl"
        path=directory/name
        if not path.is_file():
            raise ValueError(f"Prepared file missing: {path}; rerun data prepare into a new output directory")
        if manifest.get("outputs",{}).get(name) != file_hash(path):
            raise ValueError("Prepared artifact hash mismatch")
        rows=read_jsonl(path)
        for row in rows:
            if row.get("split") != split:
                raise ValueError("Prepared row split disagrees with file split")
            if section == "preferences":
                a,b=row["chosen"],row["rejected"]
                if any(a[key] != b[key] for key in ("task_id","group_id","prompt","form","mode","split","vocab_sha256")) or a["split"] != split or row["task_id"] != a["task_id"]:
                    raise ValueError("Preference candidates disagree on context/split/vocabulary")
                if a["text"] == b["text"]:
                    raise ValueError("Preference candidates are identical")
        result[split]=rows
    samples=[s for rows in result.values() for row in rows for s in ([row] if section == "sft" else [row["chosen"],row["rejected"]])]
    for sample in samples:
        validate_sample(sample)
    validate_group_splits(samples)
    bytext={}
    for sample in samples:
        if sample["text"] in bytext and bytext[sample["text"]] != sample["split"]:
            raise ValueError("Identical poem across training and validation")
        bytext[sample["text"]]=sample["split"]
    if require_ready:
        groups = {split: rows if section == "sft" else [r["chosen"] for r in rows]
                  for split, rows in result.items()}
        report = readiness(groups)
        if not report["ready"]:
            details = "; ".join(f"{split}: {report['counts'][split]} usable, missing {', '.join(report['missing'][split]) or 'none'}"
                                for split in ("train", "val"))
            raise ValueError(f"Prepared {section} data is not trainable: {details}. "
                             "Increase/calibrate qualified candidates and prepare a new dataset; training was not started.")
    return result,manifest



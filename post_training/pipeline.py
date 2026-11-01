"""Resumable scoring, preparation and optional RSFT for completed candidates."""
import argparse
import json
from .core import resolve, read_complete_samples, read_jsonl, load_config
from . import data, judge
from .prepared import load_prepared, readiness


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--scores", help="Score file; defaults to candidates/scores_batched_v1.jsonl")
    parser.add_argument("--output", help="New training output directory")
    parser.add_argument("--config", default="configs/post_training_v3.json")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true", help="Allow paid API scoring and preparation")
    parser.add_argument("--train", action="store_true", help="Train after all data checks pass")
    args=parser.parse_args(argv)
    if args.train and (not args.execute or not args.output):
        parser.error("--train requires --execute and --output")
    if args.train and resolve(args.output).exists():
        raise ValueError(f"Training output already exists: {resolve(args.output)}; choose a new --output")
    candidates=resolve(args.candidates)
    samples=candidates/"samples.jsonl"
    rows=read_complete_samples(samples)
    scores=resolve(args.scores) if args.scores else candidates/"scores_batched_v1.jsonl"
    cfg=load_config(args.config)
    if args.batch_size is not None:
        cfg["api"]["batch_size"]=args.batch_size
    # A complete, validated legacy score file can be used without paying for rescoring.
    complete=False
    if scores.is_file():
        labels=read_jsonl(scores)
        complete=len(data.checked_labels(rows, labels, "sample_id")) == len(rows)
    if not complete:
        result=judge.run_judging(cfg,samples,scores,"score",execute=args.execute)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if not args.execute:
            print("Dry run only. Add --execute to score and prepare.")
            return 0
        if result["remaining_items"]:
            print("Scoring incomplete; rerun the same command to resume. No preparation or training started.")
            return 1
    elif not args.execute:
        print("Scores complete. Add --execute to prepare and optionally train.")
        return 0
    data.main(["prepare","--samples",str(samples),"--scores",str(scores),"--output",args.prepared,"--config",args.config])
    prepared,_=load_prepared(args.prepared,"sft",require_ready=False)
    report=readiness(prepared)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not report["ready"]:
        print("Not enough qualified data. Increase candidates or calibrate the judge; use a new prepared path.")
        return 2
    if args.train:
        from . import sft
        return sft.main(["--data",args.prepared,"--output",args.output,"--config",args.config,"--device",args.device]) or 0
    print("Prepared data is ready. Add --train --output PATH to start RSFT.")
    return 0

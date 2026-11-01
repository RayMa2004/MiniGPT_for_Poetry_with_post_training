"""Offline matched blind-comparison artifacts and held-out quality summaries."""
import argparse
import random
from collections import defaultdict
from .core import (SPLITS, read_json, read_jsonl, write_jsonl, resolve, digest, file_hash,
                   new_output, atomic_json, unique_index, validate_sample, read_complete_samples)
from .data import checked_samples, checked_labels


def comparison_pairs(baseline, candidate, seed=42, split="test"):
    if split not in ("val","test"):
        raise ValueError("Quality comparison must use val/test tasks")
    checked_samples(baseline)
    checked_samples(candidate)
    def keyed(rows):
        result={}
        for sample in rows:
            if sample["split"] == split:
                key=(sample["task_id"],sample["sample_index"])
                if key in result:
                    raise ValueError("Duplicate task/sample_index in comparison")
                result[key]=sample
        return result
    left,right=keyed(baseline),keyed(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError("Baseline and candidate must cover exactly the same held-out tasks and sample indexes")
    rng=random.Random(seed)
    pairs,identical=[],[]
    for key,a in sorted(left.items()):
        b=right[key]
        if any(a[field] != b[field] for field in ("prompt","form","mode","split","group_id","seed","sampling","vocab_sha256")):
            raise ValueError("Comparison context/seed/decoding/vocabulary mismatch")
        if a["text"] == b["text"]:
            identical.append({"task_id":a["task_id"],"sample_index":a["sample_index"],"mode":a["mode"],"form":a["form"],
                              "baseline_sample_id":a["sample_id"],"candidate_sample_id":b["sample_id"],"text_sha256":digest(a["text"])})
            continue
        side="A" if rng.random() < .5 else "B"
        pair={"task_id":a["task_id"],"split":split,"a":b if side == "A" else a,
              "b":a if side == "A" else b,"candidate_side":side}
        pair["pair_id"]=digest(pair)[:24]
        pairs.append(pair)
    return {"pairs":pairs,"identical":identical,"split":split,
            "matched_count":len(left),"baseline_checkpoint_sha256":baseline[0]["checkpoint_sha256"],
            "candidate_checkpoint_sha256":candidate[0]["checkpoint_sha256"]}


def summarize_comparison(comparison, labels):
    pairs=comparison["pairs"]
    indexed=checked_labels(pairs,labels,"pair_id")
    if len(indexed) != len(pairs):
        raise ValueError("Comparison has missing judgments; resume labeling before reporting")
    counters=defaultdict(lambda:{"candidate_wins":0,"baseline_wins":0,"ties":0,"both_bad":0,"identical":0})
    for item in comparison["identical"]:
        for key in ("overall",item["mode"]+"/"+item["form"]):
            counters[key]["identical"]+=1
    for pair in pairs:
        winner=indexed[pair["pair_id"]]["result"].get("winner")
        if winner not in ("A","B","tie","both_bad"):
            raise ValueError("Invalid comparison winner")
        category=("candidate_wins" if winner == pair["candidate_side"] else "baseline_wins") if winner in ("A","B") else ("ties" if winner == "tie" else "both_bad")
        sample=pair["a"]
        for key in ("overall",sample["mode"]+"/"+sample["form"]):
            counters[key][category]+=1
    expected={mode+"/"+form for mode in ("prefix","free") for form in ("wujue","qijue")}
    if expected-set(counters):
        raise ValueError("Final comparison coverage missing required mode/form groups")
    for values in counters.values():
        total=sum(values.values())
        decisive=values["candidate_wins"]+values["baseline_wins"]
        values.update(count=total,candidate_win_rate=values["candidate_wins"]/total,
                      decisive_win_rate=values["candidate_wins"]/decisive if decisive else None,
                      both_bad_rate=values["both_bad"]/total)
    return dict(counters,status="complete",split=comparison["split"],matched_count=comparison["matched_count"],
                interpretation="AI-judged matched samples, not human ground truth. Repeated empty-prefix samples do not establish unseen-prompt generalization; inspect real human blind ratings.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    build=sub.add_parser("build")
    build.add_argument("--baseline",required=True)
    build.add_argument("--candidate",required=True)
    build.add_argument("--output",required=True)
    build.add_argument("--split",choices=("val","test"),default="test")
    build.add_argument("--seed",type=int,default=42)
    report=sub.add_parser("report")
    report.add_argument("--comparison",required=True)
    report.add_argument("--preferences")
    report.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command == "build":
        result=comparison_pairs(read_complete_samples(resolve(args.baseline)),read_complete_samples(resolve(args.candidate)),args.seed,args.split)
        out=new_output(args.output)
        write_jsonl(out/"pairs.jsonl",result["pairs"])
        atomic_json(out/"comparison.json",result)
        atomic_json(out/"manifest.json",{"baseline":file_hash(resolve(args.baseline)),"candidate":file_hash(resolve(args.candidate)),"comparison_sha256":file_hash(out/"comparison.json")})
        print(f"Matched={result['matched_count']}; API pairs={len(result['pairs'])}; exact ties={len(result['identical'])}")
    else:
        comparison=read_json(resolve(args.comparison))
        report=summarize_comparison(comparison,read_jsonl(resolve(args.preferences)) if args.preferences else [])
        out=new_output(args.output)
        atomic_json(out/"report.json",report)
        print(report)


if __name__ == "__main__":
    main()

"""Task splitting, paired comparisons and quality-gated RSFT artifacts (no torch)."""
import argparse
import json
import itertools
import random
import re
from collections import Counter, defaultdict
from .core import (DIMENSIONS, SPLITS, FORMS, ROOT, resolve, read_json, read_jsonl,
                   write_jsonl, digest, file_hash, atomic_json, new_output, artifact_path,
                   load_config, validate_task, validate_sample, unique_index, validate_group_splits, read_complete_samples)


def partition_groups(names, seed):
    names = sorted(set(names))
    if len(names) < 10:
        raise ValueError("每类任务至少10个组，才能建立独立train/val/test候选池")
    random.Random(seed).shuffle(names)
    n = max(1, round(len(names)*0.1))
    return {name: ("test" if i < n else "val" if i < 2*n else "train") for i, name in enumerate(names)}


def build_tasks(prefixes, free_groups=10, seed=42, namespace="main"):
    if not isinstance(namespace,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}",namespace):
        raise ValueError("Task namespace must use 1–32 letters/digits/underscore/hyphen")
    if type(free_groups) is not int or free_groups < 10:
        raise ValueError("free_groups must be integer >= 10")
    if not isinstance(prefixes, list) or any(not isinstance(p, str) or not p for p in prefixes):
        raise ValueError("prefixes must be nonempty strings")
    prefixes = sorted(set(prefixes))
    prefix_split = partition_groups(prefixes, seed)
    free_split = partition_groups([f"free:{namespace}:{i}" for i in range(free_groups)], seed+1)
    tasks = []
    for prompt, split in prefix_split.items():
        for form in FORMS:
            item = {"pool":namespace,"group_id": "prefix:"+prompt, "prompt": prompt, "form": form, "mode": "prefix", "split": split}
            item["task_id"] = digest(item)[:24]
            tasks.append(validate_task(item))
    for group, split in free_split.items():
        for form in FORMS:
            item = {"pool":namespace,"group_id": group, "prompt": "", "form": form, "mode": "free", "split": split}
            item["task_id"] = digest(item)[:24]
            tasks.append(validate_task(item))
    validate_group_splits(tasks)
    return sorted(tasks, key=lambda t: t["task_id"])


def checked_samples(samples):
    unique_index(samples, "sample_id")
    for sample in samples:
        validate_sample(sample)
    validate_group_splits(samples)
    for key in ("checkpoint_sha256", "vocab_sha256"):
        if len({row[key] for row in samples}) > 1:
            raise ValueError(f"一次数据准备仅接受同一来源模型和词表: {key}")
    tasks = {}
    for sample in samples:
        context = {key: sample[key] for key in ("group_id", "prompt", "form", "mode", "split")}
        if sample["task_id"] in tasks and tasks[sample["task_id"]] != context:
            raise ValueError("task_id refers to inconsistent contexts")
        tasks[sample["task_id"]] = context
    return samples


def build_pairs(samples, max_per_task=4, seed=42):
    if type(max_per_task) is not int or max_per_task < 1:
        raise ValueError("max_per_task must be positive integer")
    checked_samples(samples)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["task_id"]].append(sample)
    pairs = []
    rng = random.Random(seed)
    for task_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["sample_id"])
        # Collapse exact repeated candidates before constructing comparisons.
        rows = list({row["text"]: row for row in rows}.values())
        candidates = list(itertools.combinations(rows, 2))
        rng.shuffle(candidates)
        for a, b in candidates[:max_per_task]:
            if rng.random() < 0.5:
                a, b = b, a
            pair = {"task_id":task_id, "split":a["split"], "a":a, "b":b}
            pair["pair_id"] = digest(pair)[:24]
            pairs.append(pair)
    return pairs


def checked_labels(items, labels, id_key):
    sources = unique_index(items, id_key)
    result = {}
    judges = set()
    for label in labels:
        item_id = label["item_id"]
        if item_id not in sources or label.get("item_hash") != digest(sources[item_id]):
            raise ValueError("评分引用已改变/不存在的样本，拒绝使用缓存分数")
        if item_id in result:
            raise ValueError("同一条目有重复评分")
        if label.get("status") != "ok":
            raise ValueError("评分输入含失败记录；请仅传入成功评分文件")
        if not label.get("judge_id"):
            raise ValueError("评分缺少judge_id")
        judges.add(label["judge_id"])
        result[item_id] = label
    if len(judges) > 1:
        raise ValueError("不能混用不同模型/规则版本的评分")
    return result


def validate_scores(result):
    scores = result.get("scores", {})
    if set(scores) != set(DIMENSIONS) or any(type(v) is not int or not 1 <= v <= 5 for v in scores.values()):
        raise ValueError("评分必须为四项1–5整数")
    return scores


def prepare(samples, score_labels, pairs, pair_labels, selection):
    checked_samples(samples)
    index = unique_index(samples,"sample_id")
    scores = checked_labels(samples,score_labels,"sample_id")
    pair_index = unique_index(pairs,"pair_id")
    for pair in pairs:
        a,b=pair["a"],pair["b"]
        if any(s.get("sample_id") not in index or digest(index[s["sample_id"]]) != digest(s) for s in (a,b)):
            raise ValueError("配对样本与采样文件不一致")
        if a["task_id"] != b["task_id"] or pair["task_id"] != a["task_id"] or pair["split"] != a["split"]:
            raise ValueError("配对混用不同任务")
        if a["text"] == b["text"]:
            raise ValueError("相同文本不能构成偏好对")
    preferences = checked_labels(pairs,pair_labels,"pair_id")
    # If one text occurs across splits, reserve it for the most held-out split.
    owners = {}
    priority = {"test":0,"val":1,"train":2}
    for sample in samples:
        text=sample["text"]
        if text not in owners or priority[sample["split"]] < priority[owners[text]]:
            owners[text]=sample["split"]
    minimum_mean=selection["minimum_mean"]
    minimum_dimension=selection["minimum_dimension"]
    maximum=selection["max_per_task"]
    if not 1 <= minimum_mean <= 5 or not 1 <= minimum_dimension <= 5 or type(maximum) is not int or maximum < 1:
        raise ValueError("Invalid selection thresholds")
    candidates=defaultdict(list)
    for item_id,label in scores.items():
        sample=index[item_id]
        values=validate_scores(label["result"])
        mean=sum(values.values())/len(DIMENSIONS)
        if owners[sample["text"]] == sample["split"] and mean >= minimum_mean and min(values.values()) >= minimum_dimension:
            candidates[sample["task_id"]].append((mean,sample,label))
    sft={split:[] for split in SPLITS}
    selected_texts=set()
    for task_id, rows in sorted(candidates.items()):
        used=0
        for mean,sample,label in sorted(rows,key=lambda row:(-row[0],row[1]["sample_id"])):
            if sample["text"] in selected_texts:
                continue
            sft[sample["split"]].append(dict(sample,judge_id=label["judge_id"],quality_mean=mean,judge_scores=label["result"]["scores"]))
            selected_texts.add(sample["text"])
            used+=1
            if used >= maximum:
                break
    output_preferences={split:[] for split in SPLITS}
    seen_pairs=set()
    for pair_id,label in preferences.items():
        pair=pair_index[pair_id]
        winner=label["result"].get("winner")
        if winner not in ("A","B","tie","both_bad"):
            raise ValueError("Invalid preference winner")
        if winner in ("tie","both_bad"):
            continue
        chosen,rejected=(pair["a"],pair["b"]) if winner == "A" else (pair["b"],pair["a"])
        if any(owners[s["text"]] != pair["split"] for s in (chosen,rejected)):
            continue
        identity=tuple(sorted((chosen["text"],rejected["text"])))
        if identity in seen_pairs:
            continue
        seen_pairs.add(identity)
        output_preferences[pair["split"]].append({"pair_id":pair_id,"task_id":pair["task_id"],
            "split":pair["split"],"chosen":chosen,"rejected":rejected,"judge_id":label["judge_id"],
            "judge_reason":label["result"].get("reason","")})
    def coverage(rows):
        return dict(Counter(f'{s["mode"]}/{s["form"]}' for s in rows))
    summary={"input_samples":len(samples),"scored_samples":len(scores),"scored_pairs":len(preferences),
             "sft":{split:{"count":len(rows),"coverage":coverage(rows)} for split,rows in sft.items()},
             "preferences":{split:{"count":len(rows),"coverage":coverage([r["chosen"] for r in rows])} for split,rows in output_preferences.items()},
             "unconditional_note":"空开头没有未见prompt泛化；此处按固定采样任务/seed划分候选池并排除跨集合完全重复文本。"}
    from .prepared import readiness
    summary["readiness"] = {"sft": readiness(sft), "preferences": readiness({
        split: [row["chosen"] for row in rows] for split, rows in output_preferences.items()})}
    return {"sft":sft,"preferences":output_preferences,"summary":summary}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    tasks=commands.add_parser("tasks")
    tasks.add_argument("--prefixes",default="post_training/prompts.json")
    tasks.add_argument("--free-groups",type=int,default=20)
    tasks.add_argument("--seed",type=int,default=42)
    tasks.add_argument("--namespace",default="main",help="Separate calibration and formal candidate seed pools")
    tasks.add_argument("--output",required=True)
    pairs=commands.add_parser("pairs")
    pairs.add_argument("--samples",required=True)
    pairs.add_argument("--max-per-task",type=int,default=4)
    pairs.add_argument("--seed",type=int,default=42)
    pairs.add_argument("--output",required=True)
    prep=commands.add_parser("prepare")
    prep.add_argument("--samples",required=True)
    prep.add_argument("--scores")
    prep.add_argument("--pairs")
    prep.add_argument("--preferences")
    prep.add_argument("--config")
    prep.add_argument("--output",required=True)
    prep.add_argument("--allow-partial-labels",action="store_true",help="Explicitly prepare a labeled subset; recorded in manifest")
    check=commands.add_parser("check",help="Check prepared hashes and train/val coverage without torch or API")
    check.add_argument("--data",required=True)
    check.add_argument("--section",choices=("sft","preferences"),default="sft")
    args=parser.parse_args(argv)
    if args.command == "check":
        from .prepared import load_prepared, readiness
        rows,_=load_prepared(args.data,args.section,require_ready=False)
        report=readiness(rows if args.section == "sft" else {
            split:[r["chosen"] for r in values] for split,values in rows.items()})
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if report["ready"] else 2
    if args.command == "tasks":
        rows=build_tasks(read_json(resolve(args.prefixes)),args.free_groups,args.seed,args.namespace)
        out=new_output(args.output)
        write_jsonl(out/"tasks.jsonl",rows)
        atomic_json(out/"manifest.json",{"seed":args.seed,"namespace":args.namespace,"task_count":len(rows),"tasks_sha256":file_hash(out/"tasks.jsonl"),
                   "split_policy":"prefix_group_and_independent_unconditional_pools","source_prefixes_sha256":file_hash(resolve(args.prefixes))})
        print(f"Tasks: {len(rows)} -> {out}")
    elif args.command == "pairs":
        rows=build_pairs(read_complete_samples(resolve(args.samples)),args.max_per_task,args.seed)
        write_jsonl(artifact_path(args.output),rows)
        print(f"Pairs: {len(rows)}")
    else:
        if not args.scores and not args.preferences:
            parser.error("prepare requires --scores for SFT or --pairs and --preferences for reward data")
        if bool(args.pairs) != bool(args.preferences):
            parser.error("--pairs and --preferences must be supplied together")
        config=load_config(args.config)
        inputs={key:resolve(getattr(args,key)) for key in ("samples","scores","pairs","preferences") if getattr(args,key)}
        result=prepare(read_complete_samples(inputs["samples"]),read_jsonl(inputs["scores"]) if "scores" in inputs else [],
                       read_jsonl(inputs["pairs"]) if "pairs" in inputs else [],read_jsonl(inputs["preferences"]) if "preferences" in inputs else [],config["selection"])
        score_complete=("scores" not in inputs or result["summary"]["scored_samples"] == result["summary"]["input_samples"])
        pair_complete=("pairs" not in inputs or result["summary"]["scored_pairs"] == len(read_jsonl(inputs["pairs"])))
        labeling_complete=score_complete and pair_complete
        if not labeling_complete and not args.allow_partial_labels:
            raise ValueError("评分未完成；继续原命令补齐评分，或显式--allow-partial-labels记录子集实验")
        result["summary"]["labeling_status"]="complete" if labeling_complete else "explicit_partial_subset"
        out=artifact_path(args.output)
        identity={"inputs":{key:file_hash(path) for key,path in inputs.items()}, "selection":config["selection"]}
        if out.exists():
            if not (out/"manifest.json").is_file():
                raise ValueError(f"Incomplete prepared directory: {out}; use a new --output path, do not create manifest.json manually")
            existing=read_json(out/"manifest.json")
            if any(existing.get(key) != value for key,value in identity.items()):
                raise ValueError(f"Prepared inputs/config changed: {out}; use a new --output path")
            expected_outputs={f"{section}/{split}.jsonl" for section in ("sft","preferences") for split in SPLITS}
            if set(existing.get("outputs",{})) != expected_outputs or any(
                not (out/name).is_file() or file_hash(out/name) != expected
                for name,expected in existing.get("outputs",{}).items()):
                raise ValueError(f"Prepared files changed/incomplete: {out}; use a new --output path")
            print(f"Reusing verified prepared data: {out}")
            print(json.dumps(result["summary"],ensure_ascii=False,indent=2))
            return 0
        out=new_output(args.output)
        for section in ("sft","preferences"):
            for split,rows in result[section].items():
                write_jsonl(out/section/f"{split}.jsonl",rows)
        atomic_json(out/"manifest.json",{"inputs":{key:file_hash(path) for key,path in inputs.items()},
                    "selection":config["selection"],"summary":result["summary"],
                    "outputs":{f"{section}/{split}.jsonl":file_hash(out/section/f"{split}.jsonl") for section in ("sft","preferences") for split in SPLITS}})
        print(json.dumps(result["summary"],ensure_ascii=False,indent=2))
        return 0


if __name__ == "__main__":
    main()

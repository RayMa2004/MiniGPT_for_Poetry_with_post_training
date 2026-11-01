"""Remote-only candidate generation using the unmodified v2 generator."""
import argparse
from .core import (resolve, read_jsonl, load_config, unique_index, validate_task,
                   validate_group_splits, digest, file_hash, new_output, atomic_json, now, source_snapshot)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks",required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--samples-per-task",type=int,default=8)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config")
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args(argv)
    if args.samples_per_task < 2:
        parser.error("--samples-per-task must be >= 2")
    config=load_config(args.config)
    tasks=read_jsonl(resolve(args.tasks))
    if not tasks:
        raise ValueError("Empty task set")
    unique_index(tasks,"task_id")
    validate_group_splits(tasks)
    checkpoint=resolve(args.checkpoint or config["paths"]["checkpoint"])
    vocab=resolve(config["paths"]["vocab"])
    import torch
    from generate import load_model, generate
    from config import generation_defaults
    model,tokenizer=load_model(checkpoint,vocab,torch.device(args.device))
    unknown=sorted({ch for t in tasks for ch in t["prompt"] if ch not in tokenizer.stoi})
    if unknown:
        raise ValueError(f"后训练任务先使用已知词表汉字；OOV单独作为诊断集: {unknown}")
    sampling=generation_defaults().copy()
    sampling.pop("form",None)
    out=new_output(args.output)
    metadata={"status":"running","started_at":now(),"tasks_sha256":file_hash(resolve(args.tasks)),
              "checkpoint_sha256":file_hash(checkpoint),"vocab_sha256":file_hash(vocab),
              "sampling":sampling,"source":source_snapshot(),"expected_count":len(tasks)*args.samples_per_task}
    atomic_json(out/"manifest.json",metadata)
    count=0
    try:
        import json
        with (out/"samples.jsonl").open("x",encoding="utf-8") as handle:
            for task in tasks:
                validate_task(task)
                for index in range(args.samples_per_task):
                    seed=int(digest({"task":task["task_id"],"index":index,"seed":config["seed"]})[:8],16)
                    details=generate(model,tokenizer,task["prompt"],form=task["form"],seed=seed,return_details=True,**sampling)
                    row=dict(task,sample_index=index,seed=seed,text=details["text"],continuation=details["continuation"],
                             checkpoint_sha256=metadata["checkpoint_sha256"],vocab_sha256=metadata["vocab_sha256"],
                             sampling=sampling,termination=details["termination"])
                    row["sample_id"]=digest(row)[:24]
                    handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                    handle.flush()
                    count+=1
                    print(f"Generated {count}/{metadata['expected_count']}",flush=True)
        metadata.update(status="completed",finished_at=now(),sample_count=count,samples_sha256=file_hash(out/"samples.jsonl"))
    except BaseException as error:
        metadata.update(status="failed",finished_at=now(),sample_count=count,error_type=type(error).__name__)
        raise
    finally:
        atomic_json(out/"manifest.json",metadata)


if __name__ == "__main__":
    main()

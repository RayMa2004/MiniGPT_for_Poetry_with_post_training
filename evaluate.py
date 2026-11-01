"""统一 LM 与固定提示词评估；仅远程显式运行时执行模型计算。"""
import argparse
import json
import time
from functools import partial
from pathlib import Path
from contextlib import nullcontext
from config import default_config
from project_paths import project_path, DEFAULT_CHECKPOINT, DEFAULT_PROMPTS, DEFAULT_EVALUATIONS

from evaluation.metrics import FORM_WIDTH, generation_metrics, language_model_metrics
from experiment import (Experiment, atomic_json, data_identity, fingerprint,
                        now, runtime_environment, same_identity, source_identity)


def load_prompts(path):
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("提示词必须是非空 JSON 列表")
    seen = set()
    for prompt in prompts:
        if not isinstance(prompt.get("id"), str) or not prompt["id"] or prompt["id"] in seen:
            raise ValueError("提示词 id 必须唯一且非空")
        if not isinstance(prompt.get("prompt"), str) or prompt.get("target_form") not in {None, *FORM_WIDTH}:
            raise ValueError("prompt/target_form 格式错误")
        seen.add(prompt["id"])
    return prompts


def analyze_samples(path, output):
    """只分析已有文本；EOS 字段缺失时报告未知，不根据最后标点猜测。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload if isinstance(payload, list) else payload["samples"]
    if Path(output).exists():
        raise FileExistsError(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"source": fingerprint(path), "created_at": now(), "generation": generation_metrics(samples)})


def evaluation_inputs(args):
    parent = Experiment.open(project_path(args.run_dir)) if args.run_dir else None
    parent_cfg = json.loads((parent.path / 'config.json').read_text(encoding='utf-8')) if parent else default_config()
    if args.checkpoint:
        checkpoint = project_path(args.checkpoint)
    elif parent:
        reference = parent_cfg.get('checkpoint_reference')
        checkpoint = project_path(reference['path']) if reference else parent.checkpoint_path('best')
    else:
        checkpoint = DEFAULT_CHECKPOINT
    paths = {'vocab': project_path(args.vocab or parent_cfg['data']['vocab']),
             'val': project_path(args.val or parent_cfg['data']['val'])}
    test = args.test or parent_cfg['data'].get('test')
    if test:
        paths['test'] = project_path(test)
    return parent, parent_cfg, checkpoint, paths


def run_evaluation(args):
    import torch
    from torch.utils.data import DataLoader
    from checkpoint import load_model_checkpoint
    from dataset import PoetryDataset, collate_batch
    from generate import generate, sampling_config
    from tokenizer import CharTokenizer

    parent, parent_cfg, checkpoint_path, paths = evaluation_inputs(args)
    vocab_path = paths['vocab']
    identity = data_identity(paths)
    expected = parent_cfg.get("data_identity", {})
    common = expected.keys() & identity.keys()
    if not same_identity({k: expected[k] for k in common}, {k: identity[k] for k in common}):
        raise ValueError("评估词表/数据与父实验登记指纹不一致；创建独立实验记录新数据")
    cp_identity = fingerprint(checkpoint_path)
    reference = parent_cfg.get("checkpoint_reference")
    if reference and cp_identity["sha256"] != reference["sha256"]:
        raise ValueError("指定权重与已登记 Baseline 内容不同")
    prompts = load_prompts(args.prompts)
    if args.samples_per_prompt < 1 or args.batch_size < 1:
        raise ValueError("samples_per_prompt/batch_size 必须 > 0")
    canonical_policy = (next((name for name in ('best', 'last')
                              if checkpoint_path == parent.checkpoint_path(name).resolve()), None) if parent
                        else 'best' if checkpoint_path == DEFAULT_CHECKPOINT else None)
    parent_run_id = parent.metadata['run_id'] if parent else None
    policy = args.checkpoint_policy or canonical_policy or ("best" if reference else "external")
    cfg = {"parent_run_id": parent_run_id, "checkpoint": cp_identity,
           "data_identity": identity, "prompts": fingerprint(args.prompts), "seed": args.seed,
           "sampling": sampling_config(args), "hard_form": args.hard_form,
           "samples_per_prompt": args.samples_per_prompt, "mode": args.mode,
           "batch_size": args.batch_size, "legacy_model_config": None,
           "source_identity": source_identity(),
           "checkpoint_policy": policy,
           "sample_seed_rule": "seed + prompt_index * samples_per_prompt + sample_index"}
    legacy = json.loads(Path(args.legacy_model_config).read_text(encoding="utf-8")) if args.legacy_model_config else None
    cfg["legacy_model_config"] = legacy
    with (parent.lock() if parent else nullcontext()):
        if fingerprint(checkpoint_path)["sha256"] != cp_identity["sha256"]:
            raise RuntimeError("获取锁前 Checkpoint 内容已变化，请重新评估")
        run = Experiment.create(parent.path / "evaluations" if parent else DEFAULT_EVALUATIONS, args.evaluation_id, args.seed, cfg,
                                args.description, "evaluation", parent_run_id=parent_run_id)
        with run.lock():
            started = time.perf_counter()
            run.update(status="running", started_at=now(), parent_run_id=parent_run_id)
            try:
                device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
                tokenizer = CharTokenizer.load(vocab_path)
                model, checkpoint = load_model_checkpoint(checkpoint_path, tokenizer, device, legacy, vocab_path)
                if parent and parent.metadata["kind"] == "training":
                    if checkpoint.get("run_id") != parent.metadata["run_id"] or checkpoint.get("config") != parent_cfg:
                        raise ValueError("指定 Checkpoint 不属于该父训练实验；外部权重应单独登记引用")
                checkpoint_identity = checkpoint.get("data_identity", {})
                common = identity.keys() & checkpoint_identity.keys()
                if not same_identity({k: identity[k] for k in common}, {k: checkpoint_identity[k] for k in common}):
                    raise ValueError("评估数据与 Checkpoint 内指纹不一致")
                model_cfg = checkpoint.get("model_config", legacy)
                progress = {"epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step")}
                run.update(environment=runtime_environment(torch, device),
                           model_parameters=sum(p.numel() for p in model.parameters()), model_config=model_cfg,
                           data_identity=identity, checkpoint=cp_identity)
                result = {"schema_version": 2, "run_id": run.metadata["run_id"],
                          "parent_run_id": parent_run_id, "config": cfg,
                          "checkpoint_progress": progress,
                          "model_config": model_cfg, "language_model": {}, "generation": None}
                # 释放优化器等 CPU 权重副本，评估只保留模型。
                del checkpoint
                if args.mode in {"all", "lm"}:
                    for split in ("val", "test"):
                        if split not in paths:
                            continue
                        dataset = PoetryDataset(paths[split], tokenizer, model.block_size)
                        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                            collate_fn=partial(collate_batch, pad_id=tokenizer.pad_id))
                        result["language_model"][split] = language_model_metrics(model, loader, device)
                        result["language_model"][split]["sample_count"] = len(dataset)
                        result["language_model"][split]["block_size"] = model.block_size
                if args.mode in {"all", "generation"}:
                    samples = []
                    for p_index, prompt in enumerate(prompts):
                        for sample_index in range(args.samples_per_prompt):
                            seed = args.seed + p_index * args.samples_per_prompt + sample_index
                            hard_form = prompt.get("target_form") if args.hard_form else None
                            sample = generate(model, tokenizer, prompt["prompt"], seed=seed, form=hard_form,
                                              return_details=True, **cfg["sampling"])
                            sample.update(prompt_id=prompt["id"], prompt=prompt["prompt"],
                                          target_form=prompt.get("target_form"), sample_index=sample_index,
                                          checkpoint=cp_identity, seed=seed, sampling=cfg["sampling"],
                                          human_scores={"fluency": None, "coherence": None, "imagery": None,
                                                        "theme_relevance": None})
                            sample["metrics"] = generation_metrics([sample])
                            samples.append(sample)
                            # 持续原子落盘；失败运行保留已有样本，但不当作完整评估。
                            atomic_json(run.path / "samples.json", {"config": cfg, "samples": samples})
                    result["generation"] = generation_metrics(samples)
                result["elapsed_seconds"] = time.perf_counter() - started
                atomic_json(run.path / "evaluation.json", result)
                run.update(status="completed", ended_at=now(), elapsed_seconds=result["elapsed_seconds"])
                if parent:
                    # 新运行保留父子索引；独立评估不依赖历史实验目录。
                    index_path = parent.path / 'evaluation.json'
                    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {'evaluations': []}
                    index['evaluations'].append(str(run.path.relative_to(parent.path)))
                    atomic_json(index_path, index)
                print(run.path)
            except BaseException as error:
                run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", ended_at=now(),
                           elapsed_seconds=time.perf_counter() - started, error=f"{type(error).__name__}: {error}")
                raise


def build_parser():
    from generate import add_sampling_arguments
    parser = argparse.ArgumentParser(description="默认评估选定v2权重与data；输出到evaluations，同名拒绝覆盖")
    parser.add_argument("--run-dir", type=project_path)
    parser.add_argument("--checkpoint", type=project_path)
    parser.add_argument("--checkpoint-policy", choices=["best", "last", "external"], help="跨 Seed 分组的权重选择口径，默认从文件名判断")
    parser.add_argument("--vocab", type=project_path)
    parser.add_argument("--val", type=project_path)
    parser.add_argument("--test", type=project_path)
    parser.add_argument("--legacy-model-config", type=project_path)
    parser.add_argument("--device")
    parser.add_argument("--evaluation-id", default="quality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts", type=project_path, default=DEFAULT_PROMPTS)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=["all", "lm", "generation"], default="all")
    parser.add_argument("--hard-form", action="store_true", help="仅对有 target_form 的提示启用硬约束")
    parser.add_argument("--description", default="")
    parser.add_argument("--samples-input", type=project_path, help="只分析已有samples JSON，不加载模型")
    parser.add_argument("--output", type=project_path, help="--samples-input的全新输出文件")
    add_sampling_arguments(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.samples_input:
        if not args.output:
            parser.error("--samples-input 需要 --output")
        analyze_samples(args.samples_input, args.output)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()

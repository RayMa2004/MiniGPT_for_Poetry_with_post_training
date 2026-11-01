"""实验训练入口；导入时不创建模型、目录或 DataLoader。仅在远程运行。"""
import argparse
import json
import math
import random
import signal
import time
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from checkpoint import (FORMAT_VERSION, capture_rng, restore_rng, read_checkpoint,
                        save_checkpoint, optimizer_to, load_model_checkpoint)
from config import load_config
from project_paths import project_path, DEFAULT_RUNS
from dataset import PoetryDataset, collate_batch
from evaluation.metrics import language_model_metrics
from experiment import Experiment, data_identity, same_identity, runtime_environment, now, source_identity
from model import MiniGPT
from scheduler import WarmupCosine
from tokenizer import CharTokenizer


def evaluate(model, loader):
    """兼容旧 train.evaluate；空目标集返回 NaN，否则返回 Token 加权 Loss。"""
    loss = language_model_metrics(model, loader)["loss"]
    return float("nan") if loss is None else loss


def validate_config(cfg):
    t = cfg["training"]
    if not isinstance(cfg["seed"], int) or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed 必须为 [0,2**32) 内整数，以兼容 NumPy")
    if t["batch_size"] < 1 or t["max_epochs"] < 1 or t["log_every"] < 1:
        raise ValueError("batch_size/max_epochs/log_every 必须 > 0")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0 or t["grad_clip"] <= 0:
        raise ValueError("learning_rate/grad_clip 必须 > 0，weight_decay 必须非负")
    if t["max_steps"] is not None and t["max_steps"] < 1:
        raise ValueError("max_steps 必须为正数或 null")
    if t["scheduler"] not in {"constant", "cosine"}:
        raise ValueError("scheduler 必须为 constant/cosine")


def train_run(run, cfg, device, resume=None, init_checkpoint=None):
    t = cfg["training"]
    validate_config(cfg)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    if t["deterministic"]:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    runtime_data = {key: project_path(value) if value is not None else None for key, value in cfg["data"].items()}
    tokenizer = CharTokenizer.load(runtime_data["vocab"])
    identity = data_identity(runtime_data)
    if not same_identity(cfg["data_identity"], identity):
        raise ValueError("数据或词表指纹变化，拒绝续训/启动")
    if not same_identity(cfg["source_identity"], source_identity(cfg.get("source_scope", "all"))):
        raise ValueError("源码与实验快照不一致；使用新实验记录代码版本变化")
    datasets = {name: PoetryDataset(runtime_data[name], tokenizer, cfg["model"]["block_size"])
                for name in ("train", "val")}
    if any(len(ds) == 0 for ds in datasets.values()):
        raise ValueError("训练或验证数据没有可用样本")
    loaders = {name: DataLoader(ds, batch_size=t["batch_size"], shuffle=name == "train",
                                num_workers=0, pin_memory=device.type == "cuda",
                                collate_fn=partial(collate_batch, pad_id=tokenizer.pad_id))
               for name, ds in datasets.items()}
    model_config = dict(cfg["model"], vocab_size=tokenizer.vocab_size)
    if init_checkpoint:
        model, source = load_model_checkpoint(init_checkpoint, tokenizer, device, vocab_path=runtime_data["vocab"])
        effective = source["model_config"]
        if effective != model_config:
            raise ValueError("热启动模型配置与新实验不一致")
        del source
    else:
        model = MiniGPT(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    total_steps = min(t["max_epochs"] * len(loaders["train"]), t["max_steps"] or math.inf)
    scheduler = WarmupCosine(optimizer, t["learning_rate"], total_steps, t["warmup_steps"], t["min_lr"]) if t["scheduler"] == "cosine" else None
    start_epoch, global_step, tokens_seen, elapsed_before = 1, 0, 0, 0.0
    best_loss, best_step = float("inf"), None
    rng_to_restore = None
    if resume:
        cp = read_checkpoint(resume)
        required = {"optimizer_state_dict", "scheduler_state_dict", "rng_state", "global_step",
                    "tokens_seen", "best_val_loss", "best_step", "epoch", "epoch_complete",
                    "run_id", "config", "data_identity", "elapsed_seconds"}
        if cp.get("format_version") != FORMAT_VERSION or required - cp.keys():
            raise ValueError("旧版/不完整 Checkpoint 不能精确恢复；用 --init-checkpoint 新实验热启动")
        if cp["run_id"] != run.metadata["run_id"] or cp["config"] != cfg or not same_identity(cp["data_identity"], identity):
            raise ValueError("Checkpoint 与运行配置/数据/Run ID 不一致")
        if cp["model_config"] != model_config or not (cp["epoch_complete"] or cp.get("budget_complete")):
            raise ValueError("只支持相同架构的完整 Epoch 边界恢复")
        if cp.get("device_type") != device.type:
            raise ValueError("状态恢复要求相同设备类型；跨设备请使用新实验热启动")
        model.load_state_dict(cp["model_state_dict"], strict=True)
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        optimizer_to(optimizer, device)
        if scheduler:
            if cp["scheduler_state_dict"] is None:
                raise ValueError("缺少 Scheduler 状态")
            scheduler.load_state_dict(cp["scheduler_state_dict"])
            if scheduler.completed_steps != cp["global_step"]:
                raise ValueError("Scheduler 步数与 global_step 不一致")
        elif cp["scheduler_state_dict"] is not None:
            raise ValueError("固定 LR 实验却包含 Scheduler 状态")
        start_epoch, global_step, tokens_seen = cp["epoch"] + 1, cp["global_step"], cp["tokens_seen"]
        best_loss, best_step = cp["best_val_loss"], cp["best_step"]
        elapsed_before = cp["elapsed_seconds"]
        rng_to_restore = cp["rng_state"]
        run.reconcile_metrics(global_step)
        if best_step is not None:
            best_path = run.checkpoint_path("best")
            if best_path.exists():
                best_cp = read_checkpoint(best_path)
                if best_cp.get("global_step") != best_step and best_step != global_step:
                    raise ValueError("best.pt 与 last.pt 不一致，需人工检查")
                del best_cp
            if best_step == global_step:
                save_checkpoint(best_path, cp)
            elif not best_path.exists():
                raise FileNotFoundError("缺少历史 best.pt，无法恢复最佳模型记录")
        del cp
    run.update(status="running", started_at=run.metadata.get("started_at") or now(), ended_at=None,
               error=None, interrupted_step=None, global_step=global_step, tokens_seen=tokens_seen,
               environment=runtime_environment(torch, device), model_parameters=sum(p.numel() for p in model.parameters()),
               data_identity=identity, best_val_loss=best_loss, best_step=best_step,
               deterministic=t["deterministic"], budget_steps=total_steps,
               throughput_definition="有效训练目标 token / 数据加载与训练的区间墙钟时间；不含验证、保存及日志写入",
               resume_note="仅完整 Epoch 边界；中途失败重放上一个边界之后的 Epoch")
    writer = None
    started = time.perf_counter()
    peak = None

    def elapsed():
        return elapsed_before + time.perf_counter() - started

    def payload(epoch, val_loss=None, epoch_complete=True, train_loss=None):
        return {"format_version": FORMAT_VERSION, "run_id": run.metadata["run_id"],
                "model_state_dict": model.state_dict(), "model_config": model_config,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": epoch, "epoch_complete": epoch_complete,
                "budget_complete": global_step >= total_steps, "global_step": global_step,
                "tokens_seen": tokens_seen, "best_val_loss": best_loss, "best_step": best_step,
                "val_loss": val_loss, "train_loss": train_loss,
                "elapsed_seconds": elapsed(), "rng_state": capture_rng(), "config": cfg,
                "data_identity": identity, "device_type": device.type,
                "runtime_environment": runtime_environment(torch, device)}

    try:
        if t["tensorboard"]:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(run.path / "tensorboard"), purge_step=global_step + 1 if resume else None)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if rng_to_restore is not None:
            # 放在 Writer 等基础设施初始化之后，避免其潜在随机调用改变训练随机序列。
            restore_rng(rng_to_restore)
        if resume is None:
            save_checkpoint(run.checkpoint_path(), payload(0))
        for epoch in range(start_epoch, t["max_epochs"] + 1):
            if global_step >= total_steps:
                break
            model.train()
            epoch_loss, epoch_tokens, interval_tokens, interval_loss = 0.0, 0, 0, 0.0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            interval_start = time.perf_counter()
            for batch_step, (x, y) in enumerate(loaders["train"], 1):
                valid = int((y != -100).sum().item())
                if not valid:
                    continue
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"非有限 loss: epoch={epoch}, batch={batch_step}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"], error_if_nonfinite=True)
                used_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                global_step += 1
                if scheduler:
                    scheduler.step()
                tokens_seen += valid
                value = loss.item()
                epoch_loss += value * valid
                epoch_tokens += valid
                interval_loss += value * valid
                interval_tokens += valid
                is_last = batch_step == len(loaders["train"]) or global_step >= total_steps
                if global_step % t["log_every"] == 0 or is_last:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        peak = torch.cuda.max_memory_allocated(device) / 1024**2
                    duration = time.perf_counter() - interval_start
                    row = {"epoch": epoch, "global_step": global_step, "tokens_seen": tokens_seen,
                           "train_loss": interval_loss / interval_tokens, "lr": used_lr,
                           "elapsed_seconds": elapsed(), "tokens_per_second": interval_tokens / duration if duration > 0 else None,
                           "peak_gpu_memory_mb": peak, "gradient_norm": float(grad_norm.item()), "best_val_loss": best_loss,
                           "interval_tokens": interval_tokens, "interval_seconds": duration}
                    run.log_metrics(row)
                    if writer:
                        for tag, key in {"Loss/train": "train_loss", "LR": "lr", "Throughput/tokens_per_sec": "tokens_per_second",
                                         "GPU/max_memory_MB": "peak_gpu_memory_mb", "GradientNorm": "gradient_norm",
                                         "Epoch": "epoch", "GlobalStep": "global_step", "TokensSeen": "tokens_seen"}.items():
                            if row[key] is not None:
                                writer.add_scalar(tag, row[key], global_step)
                    print(f"{run.metadata['run_id']} epoch={epoch} step={global_step} loss={row['train_loss']:.4f}")
                    interval_tokens, interval_loss = 0, 0.0
                    interval_start = time.perf_counter()
                if global_step >= total_steps:
                    break
            if not epoch_tokens:
                raise RuntimeError("本 Epoch 无有效目标 token")
            result = language_model_metrics(model, loaders["val"], device)
            val_loss = result["loss"]
            if val_loss is None:
                raise RuntimeError("验证集无有效目标 token")
            improved = val_loss < best_loss
            if improved:
                best_loss, best_step = val_loss, global_step
            run.log_metrics({"epoch": epoch, "global_step": global_step, "val_loss": val_loss,
                             "val_ppl": result["ppl"], "best_val_loss": best_loss})
            if writer:
                writer.add_scalar("Loss/val", val_loss, global_step)
                writer.add_scalar("Loss/train_epoch", epoch_loss / epoch_tokens, global_step)
                if result["ppl"] is not None:
                    writer.add_scalar("Perplexity/val", result["ppl"], global_step)
                writer.add_scalar("BestValLoss", best_loss, global_step)
                writer.flush()
            cp = payload(epoch, val_loss, epoch_complete=batch_step == len(loaders["train"]),
                         train_loss=epoch_loss / epoch_tokens)
            # last 先提交；若保存 best 时崩溃，可从该 last 重建 best。
            save_checkpoint(run.checkpoint_path(), cp)
            if improved:
                save_checkpoint(run.checkpoint_path("best"), cp)
            del cp
            run.update(best_val_loss=best_loss, best_step=best_step, elapsed_seconds=elapsed(),
                       global_step=global_step, tokens_seen=tokens_seen)
        run.update(status="completed", ended_at=now(), elapsed_seconds=elapsed())
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                   ended_at=now(), elapsed_seconds=elapsed(), error=f"{type(error).__name__}: {error}",
                   interrupted_step=global_step)
        raise
    finally:
        if writer:
            writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description="MiniGPT v2训练；默认当前最佳配置与data")
    parser.add_argument("--experiment-id", default="minigpt_v2")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=project_path, help="可选JSON覆盖；默认configs/minigpt_v2.json")
    parser.add_argument("--root", type=project_path, default=DEFAULT_RUNS)
    parser.add_argument("--description", default="")
    parser.add_argument("--device")
    parser.add_argument("--resume", type=project_path, help="恢复当前源码创建的运行自己的last.pt")
    parser.add_argument("--init-checkpoint", type=project_path, help="仅权重热启动，新建运行并重置训练状态")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Ubuntu 作业管理器的 SIGTERM 也进入异常收尾；SIGKILL 只能依赖边界 Checkpoint。
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"收到信号 {signum}")
    signal.signal(signal.SIGTERM, terminate)
    if args.resume and (args.config or args.seed is not None or args.init_checkpoint):
        parser.error("恢复使用原配置，不允许 --config/--seed/--init-checkpoint")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.resume:
        resume = Path(args.resume).resolve()
        run = Experiment.open(resume.parent)
        if resume != run.checkpoint_path():
            parser.error("--resume 只接受同一运行目录的 last.pt")
        cfg = json.loads((run.path / "config.json").read_text(encoding="utf-8"))
        if run.metadata["kind"] != "training":
            parser.error("只允许恢复 training 实验")
    else:
        cfg = load_config(args.experiment_id, args.config, args.seed)
        validate_config(cfg)
        cfg["data_identity"] = data_identity({key: project_path(value) if value is not None else None
                                                   for key, value in cfg["data"].items()})
        cfg["source_scope"] = "training"
        cfg["source_identity"] = source_identity("training")
        if args.init_checkpoint:
            from experiment import fingerprint
            cfg["initial_checkpoint"] = fingerprint(args.init_checkpoint)
        run = Experiment.create(args.root, args.experiment_id, cfg["seed"], cfg, args.description)
    with run.lock():
        try:
            train_run(run, cfg, device, args.resume, args.init_checkpoint)
        except BaseException as error:
            if run.metadata["status"] not in {"failed", "interrupted"}:
                run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           ended_at=now(), error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()

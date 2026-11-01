"""实验文件管理；本模块不导入 PyTorch，不加载模型。"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import socket
import tempfile
from importlib.metadata import PackageNotFoundError, version
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from project_paths import project_path

METRIC_FIELDS = ["epoch", "global_step", "tokens_seen", "train_loss", "val_loss", "val_ppl",
                 "lr", "elapsed_seconds", "tokens_per_second", "peak_gpu_memory_mb",
                 "gradient_norm", "best_val_loss", "interval_tokens", "interval_seconds"]


def now():
    return datetime.now(timezone.utc).isoformat()


def clean_json(value):
    """JSON 中非有限浮点数用 null 表示，不输出非标准 NaN/Infinity。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(clean_json(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def data_identity(paths):
    return {name: fingerprint(path) for name, path in paths.items() if path is not None}


def source_identity(scope="all"):
    """新训练只跟踪实际训练依赖，避免绘图调整使续训失效；旧快照仍严格检查全部源码。"""
    root = Path(__file__).parent
    if scope == "training":
        names = ("train.py", "model.py", "config.py", "checkpoint.py", "experiment.py",
                 "dataset.py", "tokenizer.py", "scheduler.py", "project_paths.py", "evaluation/__init__.py", "evaluation/metrics.py")
        paths = [root / name for name in names]
    elif scope == "all":
        paths = sorted(root.glob("*.py")) + sorted((root / "evaluation").glob("*.py"))
    else:
        raise ValueError(f"未知源码指纹范围: {scope}")
    return data_identity({str(path.relative_to(root)): path for path in paths})


def same_identity(left, right):
    """移动服务器目录不影响身份；比较角色、字节数和内容哈希。"""
    return {k: (v["sha256"], v["bytes"]) for k, v in left.items()} == {
        k: (v["sha256"], v["bytes"]) for k, v in right.items()}


def runtime_environment(torch_module=None, device=None):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "hostname": socket.gethostname(), "pytorch": None, "cuda": None, "gpu": None,
            "gpu_memory_unavailable_reason": "未选择 CUDA 设备"}
    info["packages"] = {}
    for package in ("numpy", "tensorboard", "opencc-python-reimplemented", "matplotlib"):
        try:
            info["packages"][package] = version(package)
        except PackageNotFoundError:
            info["packages"][package] = None
    if torch_module is not None:
        info.update(pytorch=str(torch_module.__version__), cuda=torch_module.version.cuda)
        if device is not None and device.type == "cuda":
            try:
                info.update(gpu=torch_module.cuda.get_device_name(device), gpu_memory_unavailable_reason=None)
                info["gpu_total_memory_mb"] = torch_module.cuda.get_device_properties(device).total_memory / 1024**2
            except (RuntimeError, OSError) as error:
                info["gpu_memory_unavailable_reason"] = str(error)
    return info


class Experiment:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))

    @classmethod
    def create(cls, root, experiment_id, seed, config, description="", kind="training", parent_run_id=None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id):
            raise ValueError("experiment_id 只允许字母、数字、下划线、点和连字符")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        local_id = f"{experiment_id}_seed{seed}"
        run_id = f"{parent_run_id}__{local_id}" if parent_run_id else local_id
        path = root / local_id
        path.mkdir(exist_ok=False)  # 原子冲突保护：绝不清空旧目录。
        atomic_json(path / "config.json", config)
        atomic_json(path / "metadata.json", {
            "schema_version": 2, "run_id": run_id, "experiment_id": experiment_id,
            "parent_run_id": parent_run_id,
            "description": description, "seed": seed, "kind": kind, "status": "created",
            "created_at": now(), "started_at": None, "ended_at": None,
            "elapsed_seconds": None, "model_parameters": None,
            "best_val_loss": None, "best_step": None,
            "environment": runtime_environment(),
        })
        return cls(path)

    @classmethod
    def open(cls, path):
        return cls(path)

    def update(self, **values):
        self.metadata.update(values)
        atomic_json(self.path / "metadata.json", self.metadata)

    @contextmanager
    def lock(self):
        path = self.path / ".active.lock"
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "time": now()}, handle)
        try:
            yield self
        finally:
            path.unlink()

    def read_metrics(self):
        path = self.path / "metrics.csv"
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _write_metrics(self, rows):
        fd, temporary = tempfile.mkstemp(dir=self.path, suffix=".csv.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path / "metrics.csv")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def log_metrics(self, row):
        rows = self.read_metrics()
        key = int(row["global_step"])
        row = clean_json(row)
        for existing in rows:
            if int(existing["global_step"]) == key:
                existing.update(row)
                break
        else:
            rows.append(row)
        self._write_metrics(sorted(rows, key=lambda r: int(r["global_step"])))

    def reconcile_metrics(self, checkpoint_step):
        self._write_metrics([r for r in self.read_metrics() if int(r["global_step"]) <= checkpoint_step])

    def checkpoint_path(self, name="last"):
        if name not in {"last", "best"}:
            raise ValueError("只允许 last/best Checkpoint")
        return self.path / f"{name}.pt"


def unlock_stale(path):
    """远程 Ubuntu 上显式清理同主机已退出进程的锁；PID 存活或跨主机时拒绝。"""
    lock_path = Path(path).resolve() / ".active.lock"
    record = json.loads(lock_path.read_text(encoding="utf-8"))
    if os.name != "posix" or record["host"] != socket.gethostname():
        raise RuntimeError("只能在原 POSIX 主机验证陈旧锁；其它情况需人工核查")
    pid = record["pid"]
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("锁中的 PID 非法，拒绝自动处理")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        if json.loads(lock_path.read_text(encoding="utf-8")) != record:
            raise RuntimeError("锁内容发生变化，拒绝删除")
        lock_path.unlink()
    else:
        raise RuntimeError("锁对应 PID 仍存活（包括 PID 被复用）；拒绝删除")


def main():
    parser = argparse.ArgumentParser(description='运行目录维护；不加载模型')
    sub = parser.add_subparsers(dest='command', required=True)
    unlock = sub.add_parser('unlock-stale', help='仅在原POSIX主机释放已退出PID的陈旧锁')
    unlock.add_argument('--run-dir', type=project_path, required=True)
    args = parser.parse_args()
    unlock_stale(args.run_dir)


if __name__ == '__main__':
    main()

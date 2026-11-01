"""旧格式严格加载与 v2 Epoch 边界恢复；只加载可信来源的 Checkpoint。"""
import os
import inspect
import random
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

from experiment import fingerprint
from model import MiniGPT

FORMAT_VERSION = 2


def resolve_model_config(config):
    """缺省值必须来自 MiniGPT 构造器，不能用训练脚本的超参数替代。"""
    parameters = inspect.signature(MiniGPT.__init__).parameters
    unknown = config.keys() - (parameters.keys() - {"self"})
    if unknown or "vocab_size" not in config:
        raise ValueError(f"模型配置缺少 vocab_size 或含未知字段: {unknown}")
    defaults = {key: value.default for key, value in parameters.items()
                if key != "self" and value.default is not inspect.Parameter.empty}
    return dict(defaults, **config)


def capture_rng():
    # NumPy ndarray 转为普通列表，使 v2 可以 weights_only=True 安全加载。
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {
        "algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
        "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["algorithm"], np.array(n["keys"], dtype=np.uint32),
                         n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA 设备数与保存时不同，无法按原随机状态恢复")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def read_checkpoint(path):
    # 在 CPU 反序列化，随后显式迁移模型和优化器，避免 RNG ByteTensor 被迁至 GPU。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint 必须是字典")
    version = checkpoint.get("format_version", 1)
    if version not in (1, FORMAT_VERSION):
        raise ValueError(f"不支持的 Checkpoint 版本: {version}")
    return checkpoint


def load_model_checkpoint(path, tokenizer, device, legacy_model_config=None, vocab_path=None):
    checkpoint = read_checkpoint(path)
    cfg = checkpoint.get("model_config")
    if cfg is None:
        if legacy_model_config is None:
            raise ValueError("权重不含 model_config，需要显式 --legacy-model-config")
        cfg = legacy_model_config
    elif legacy_model_config is not None and resolve_model_config(cfg) != resolve_model_config(legacy_model_config):
        raise ValueError("显式模型配置与 Checkpoint 内配置不同；拒绝覆盖")
    cfg = resolve_model_config(cfg)
    if cfg["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("词表大小不匹配")
    identity = checkpoint.get("data_identity", {}).get("vocab")
    if identity and vocab_path and fingerprint(vocab_path)["sha256"] != identity["sha256"]:
        raise ValueError("词表 SHA256 不匹配（即使大小相同也不能交换 token ID）")
    if not identity:
        warnings.warn("旧 Checkpoint 没有词表指纹；请人工核实原始词表和数据版本", stacklevel=2)
    model = MiniGPT(**cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    checkpoint["model_config"] = cfg
    return model, checkpoint


def save_checkpoint(path, payload):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        # Windows _commit (fsync) requires a writable file handle.
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            # 非 capturable AdamW 的 step 标量保留在 CPU。
            if torch.is_tensor(value) and key != "step":
                state[key] = value.to(device)

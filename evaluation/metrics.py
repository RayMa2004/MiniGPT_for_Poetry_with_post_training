"""指标口径见 METRICS.md；纯文本指标不依赖 PyTorch。"""
import math
import re
import unicodedata
from collections import Counter

FORM_WIDTH = {"wujue": 5, "qijue": 7}


def is_han(text):
    # 使用运行时 Unicode 字符名称，避免宽范围正则把区间内未分配码点计为汉字。
    return len(text) == 1 and (text == "〇" or unicodedata.name(text, "").startswith(
        ("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")))


def form_compliant(text, form):
    width = FORM_WIDTH[form]
    # 四句，标点固定为 ，。 ，。；只忽略空白，不把其它字符算成汉字。
    text = re.sub(r"\s+", "", text)
    lines = re.fullmatch(r"([^，。]+)，([^，。]+)。([^，。]+)，([^，。]+)。", text)
    return bool(lines and all(len(line) == width and all(map(is_han, line)) for line in lines.groups()))


def text_metrics(texts):
    """Distinct 用语料级唯一集合；重复数逐条续写统计，不把不同样本的共用短语判为重复。"""
    sequences = [[ch for ch in text if is_han(ch)] for text in texts]
    result = {}
    for n in (1, 2, 3):
        per_sample = [Counter(tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)) for seq in sequences]
        counts = Counter()
        for counter in per_sample:
            counts.update(counter)
        total = sum(counts.values())
        if n <= 2:
            result[f"distinct_{n}"] = len(counts) / total if total else None
        if n >= 2:
            result[f"repeated_{'bigram' if n == 2 else 'trigram'}_ratio"] = (
                sum(sum(c.values()) - len(c) for c in per_sample) / total if total else None)
    result["average_length"] = sum(map(len, sequences)) / len(sequences) if sequences else None
    return result


def generation_metrics(samples):
    result = text_metrics([sample.get("continuation", sample["text"]) for sample in samples])
    result["sample_count"] = len(samples)
    eos_known = [s for s in samples if isinstance(s.get("ended_with_eos"), bool)]
    result["eos_observed_count"] = len(eos_known)
    result["eos_rate"] = sum(s["ended_with_eos"] for s in eos_known) / len(eos_known) if eos_known else None
    result["hard_constraint_rate"] = sum(s.get("termination") == "constraint" for s in samples) / len(samples) if samples else None
    for form in FORM_WIDTH:
        targeted = [s for s in samples if s.get("target_form") == form]
        result[f"{form}_count"] = len(targeted)
        result[f"{form}_compliance"] = sum(form_compliant(s["text"], form) for s in targeted) / len(targeted) if targeted else None
    return result


def perplexity(loss):
    if loss is None or not math.isfinite(loss):
        return None
    try:
        return math.exp(loss)
    except OverflowError:
        return None


def language_model_metrics(model, loader, device=None):
    import torch
    device = device or next(model.parameters()).device
    previous_mode = model.training
    total_loss, total_tokens = 0.0, 0
    model.eval()
    try:
        with torch.inference_mode():
            for x, y in loader:
                valid = int((y != -100).sum().item())
                if valid == 0:
                    continue
                _, loss = model(x.to(device), y.to(device))
                value = loss.item()
                if not math.isfinite(value):
                    raise RuntimeError("评估出现非有限 Loss")
                total_loss += value * valid
                total_tokens += valid
    finally:
        model.train(previous_mode)
    loss = total_loss / total_tokens if total_tokens else None
    return {"loss": loss, "ppl": perplexity(loss), "valid_tokens": total_tokens,
            "undefined_reason": None if total_tokens else "没有有效目标 Token"}

"""唐诗 v2 保守清洗纯函数；导入不读写文件，不依赖 OpenCC 或模型。"""
import math
import re
import unicodedata

PAIRS = {"(": ")", "（": "）", "[": "]", "［": "］", "【": "】", "〔": "〕", "〖": "〗"}
# 数字和“注”可能是脚注标号，不能自动视为正文校补字。
FOOTNOTE_HAN = frozenset("〇零一二三四五六七八九十百千万亿两壹贰叁肆伍陆柒捌玖拾佰仟注註")


def validate_rules(rules):
    """拒绝无效/未识别配置，避免规则拼写错误被静默忽略。"""
    fields = {"rule_version", "min_length", "max_length", "seed", "train_fraction",
              "annotation_patterns", "residual_source_markers"}
    if not isinstance(rules, dict) or set(rules) != fields:
        raise ValueError(f"清洗规则必须恰好包含 {sorted(fields)}")
    if not isinstance(rules["rule_version"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", rules["rule_version"]):
        raise ValueError("rule_version 必须为非空版本标识")
    for key in ("min_length", "max_length", "seed"):
        if type(rules[key]) is not int:
            raise ValueError(f"{key} 必须为整数")
    if not 1 <= rules["min_length"] <= rules["max_length"]:
        raise ValueError("长度范围必须满足 1 <= min_length <= max_length")
    fraction = rules["train_fraction"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("train_fraction 必须在 0 与 1 之间")
    for key in ("annotation_patterns", "residual_source_markers"):
        if not isinstance(rules[key], list) or not rules[key] or not all(
                isinstance(value, str) and value for value in rules[key]):
            raise ValueError(f"{key} 必须为非空字符串列表")
    try:
        for pattern in rules["annotation_patterns"]:
            re.compile(pattern)
    except re.error as error:
        raise ValueError(f"注释模式无效: {error}") from error


def normalized_key(text):
    """仅忽略空白与 Unicode 标点；用于将同文异标点版本分到同一集合。"""
    return "".join(ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


def bracket_spans(text):
    """栈解析顶层括号，可跨 paragraphs；不匹配时整条失败，绝不猜补边界。"""
    stack, spans = [], []
    for offset, ch in enumerate(text):
        if ch in PAIRS:
            stack.append((ch, offset))
        elif ch in PAIRS.values():
            if not stack or PAIRS[stack[-1][0]] != ch:
                return [], {"offset": offset, "character": ch, "kind": "unexpected_closer"}
            opener, start = stack.pop()
            if not stack:
                spans.append((start, offset + 1, opener))
    if stack:
        return [], {"offset": stack[0][1], "character": stack[0][0], "kind": "unclosed_opener"}
    return spans, None


def _editorial_character(content):
    return (len(content) == 1 and content not in FOOTNOTE_HAN and unicodedata.name(content, "").startswith(
        ("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")))


def clean_paragraphs(paragraphs, convert, rules):
    """convert 为确定性繁简转换函数；rules 先由调用方 validate_rules。

    偏移均指去空白、繁简转换后的 normalized_text；无法确定时 text=None。
    changes 记录候选文本的操作，隔离记录不会被写入 train/val。
    """
    result = {"status": "rejected", "text": None, "normalized_text": None,
              "reasons": [], "changes": [], "brackets": []}
    if not isinstance(paragraphs, list) or not all(isinstance(p, str) for p in paragraphs):
        result["reasons"] = ["invalid_paragraphs"]
        return result
    text = re.sub(r"\s+", "", convert("".join(paragraphs)))
    result["normalized_text"] = text
    spans, error = bracket_spans(text)
    if error:
        result.update(status="quarantined", reasons=["unbalanced_brackets"], bracket_error=error)
        return result

    operations, ambiguous = [], False
    for start, end, opener in spans:
        content = text[start + 1:end - 1]
        match = next((index for index, pattern in enumerate(rules["annotation_patterns"])
                      if re.search(pattern, content)), None)
        if opener in {"(", "（"} and match is not None:
            action, replacement = "remove_explicit_annotation", ""
        elif opener in {"[", "［"} and _editorial_character(content):
            action, replacement = "unwrap_editorial_character", content
        else:
            action, replacement = "review_ambiguous_bracket", None
            ambiguous = True
        result["brackets"].append({"start": start, "end": end, "content": text[start:end],
                                   "rule": action, "annotation_pattern_index": match})
        operations.append((start, end, replacement, action, match))
    if ambiguous:
        result.update(status="quarantined", reasons=["ambiguous_bracket"])
        return result

    pieces, cursor = [], 0
    for start, end, replacement, action, match in operations:
        pieces.append(text[cursor:start])
        # 独立尾注常为“诗。（见出处）。”；仅消费重复的同种句末标点。
        # 不压缩正文标点，所有消费记入同一删除跨度。
        preceding = "".join(pieces)
        if (action == "remove_explicit_annotation" and end < len(text) and preceding
                and text[end] in "。！？!?" and text[end] == preceding[-1]):
            end += 1
        pieces.append(replacement)
        result["changes"].append({"start": start, "end": end, "removed": text[start:end],
                                   "replacement": replacement, "rule": action,
                                   "annotation_pattern_index": match})
        cursor = end
    pieces.append(text[cursor:])
    cleaned = "".join(pieces)
    if any(marker in cleaned for marker in rules["residual_source_markers"]):
        result.update(status="quarantined", reasons=["residual_source_marker"], candidate_text=cleaned)
        return result
    if not rules["min_length"] <= len(cleaned) <= rules["max_length"]:
        result.update(reasons=["length_out_of_range"], candidate_text=cleaned)
        return result
    if not normalized_key(cleaned):
        result.update(reasons=["empty_normalized_key"], candidate_text=cleaned)
        return result
    result.update(status="accepted", text=cleaned)
    return result

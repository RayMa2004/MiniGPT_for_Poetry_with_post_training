"""MiniGPT v2生成：默认E05，整首五绝/七绝，显式free可自由生成。"""
import argparse
import json
import torch
from checkpoint import load_model_checkpoint
from evaluation.metrics import FORM_WIDTH, is_han, form_compliant
from config import generation_defaults
from project_paths import DEFAULT_CHECKPOINT, DEFAULT_VOCAB, project_path
from tokenizer import CharTokenizer

CHECKPOINT_PATH = DEFAULT_CHECKPOINT
VOCAB_PATH = DEFAULT_VOCAB
GENERATION_DEFAULTS = generation_defaults()
DEFAULT_PENALTY = GENERATION_DEFAULTS["repetition_penalty"]
DEFAULT_FORM = GENERATION_DEFAULTS["form"]


def load_model(checkpoint_path=CHECKPOINT_PATH, vocab_path=VOCAB_PATH, device=None, legacy_model_config=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab_path = project_path(vocab_path)
    checkpoint_path = project_path(checkpoint_path)
    tokenizer = CharTokenizer.load(vocab_path)
    model, _ = load_model_checkpoint(checkpoint_path, tokenizer, device, legacy_model_config, vocab_path)
    return model, tokenizer


def filter_logits(logits, top_k=20, top_p=0.9):
    """输入为一维词表 logits；保留越过 top-p 阈值的第一个候选。"""
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        logits[logits < torch.topk(logits, k).values[-1]] = float("-inf")
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        remove = torch.softmax(sorted_logits, dim=-1).cumsum(-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_indices[remove]] = float("-inf")
    return logits


def apply_repetition_penalty(logits, history, penalty=1.0):
    if penalty <= 0:
        raise ValueError("repetition_penalty 必须 > 0；1 表示关闭")
    if history and penalty != 1:
        ids = torch.tensor(sorted(set(history)), device=logits.device, dtype=torch.long)
        values = logits[ids]
        # 对出现过的 token 只惩罚一次，负 logits 乘惩罚，正 logits 除惩罚。
        logits[ids] = torch.where(values < 0, values * penalty, values / penalty)
    return logits


def banned_ngram_tokens(history, n):
    if n <= 0 or len(history) + 1 < n:
        return set()
    if n == 1:
        return set(history)
    prefix = tuple(history[-(n - 1):])
    return {history[i+n-1] for i in range(len(history) - n + 1)
            if tuple(history[i:i+n-1]) == prefix}


def form_slots(form):
    width = FORM_WIDTH[form]
    return [slot for punct in "，。，。" for slot in [None] * width + [punct]]


@torch.inference_mode()
def generate(model, tokenizer, prompt="", max_new_tokens=GENERATION_DEFAULTS["max_new_tokens"],
             temperature=GENERATION_DEFAULTS["temperature"], top_k=GENERATION_DEFAULTS["top_k"], top_p=GENERATION_DEFAULTS["top_p"],
             repetition_penalty=DEFAULT_PENALTY, no_repeat_ngram_size=GENERATION_DEFAULTS["no_repeat_ngram_size"], form=DEFAULT_FORM, seed=None, return_details=False):
    if temperature <= 0 or max_new_tokens < 0 or repetition_penalty <= 0 or no_repeat_ngram_size < 0:
        raise ValueError("temperature/penalty 必须为正，token/ngram 预算不得为负")
    if (top_k is not None and top_k < 0) or (top_p is not None and not 0 <= top_p <= 1):
        raise ValueError("top_k 必须非负，top_p 必须在 [0,1] 内；0 表示关闭")
    if form is not None and form not in FORM_WIDTH:
        raise ValueError("form 必须为 wujue/qijue 或 None")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    slots = form_slots(form) if form else None
    if slots is not None:
        if len(prompt) > len(slots):
            raise ValueError("prompt 超过诗体长度")
        for char, slot in zip(prompt, slots):
            if (slot is None and not is_han(char)) or (slot is not None and slot != char):
                raise ValueError("约束 prompt 必须是规范诗体的前缀（汉字和 ，。 标点）")
        if max_new_tokens < len(slots) - len(prompt) + 1:
            raise ValueError("max_new_tokens 不足以完成诗体（包括标点和 EOS）")
        for punctuation in "，。":
            if punctuation not in tokenizer.stoi:
                raise ValueError(f"词表缺少标点 {punctuation}")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not form:
        # 兼容原生成器：过长提示只保留 block_size-1 个 token。
        prompt_ids = prompt_ids[-(model.block_size - 1):] if model.block_size > 1 else []
    ids = [tokenizer.bos_id] + prompt_ids
    # 字符级编码一个原字对应一个ID，包括UNK。保留相同上下文窗口内的原字，
    # 不能用decode(prompt_ids)重建开头，否则skip_special_tokens会吞掉UNK。
    effective_prompt = prompt[-len(prompt_ids):] if prompt_ids else ""
    unknown_prompt_chars = list(dict.fromkeys(ch for ch in effective_prompt if ch not in tokenizer.stoi))
    unknown_prompt_count = sum(ch not in tokenizer.stoi for ch in effective_prompt)
    generated_ids = []
    han_ids = [idx for char, idx in tokenizer.stoi.items() if is_han(char)] if form else None
    if form and not han_ids:
        raise ValueError("词表没有可用汉字")
    ended_with_eos, termination, relaxations = False, "length", 0
    for _ in range(max_new_tokens):
        position = len(prompt) + len(generated_ids)
        forced_id = None
        if slots is not None:
            if position == len(slots):
                forced_id = tokenizer.eos_id
            elif slots[position] is not None:
                forced_id = tokenizer.stoi[slots[position]]
        if forced_id is None:
            idx = torch.tensor([ids[-model.block_size:]], dtype=torch.long, device=device)
            logits, _ = model(idx)
            scores = logits[0, -1, :].clone() / temperature
            scores[~torch.isfinite(scores)] = float("-inf")
            scores[[tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]] = float("-inf")
            if form:
                allowed = torch.zeros_like(scores, dtype=torch.bool)
                allowed[han_ids] = True
                scores[~allowed] = float("-inf")
            fallback = scores.clone()
            scores = apply_repetition_penalty(scores, ids[1:], repetition_penalty)
            banned = banned_ngram_tokens(ids[1:], no_repeat_ngram_size)
            if banned:
                scores[list(banned)] = float("-inf")
            if not torch.isfinite(scores).any():
                # 冲突时仅放宽重复控制，保留特殊 token 屏蔽与汉字约束，记录次数。
                scores = fallback
                relaxations += 1
            if not torch.isfinite(scores).any():
                raise RuntimeError("没有有效采样候选：检查词表、模型 logits 和约束")
            scores = filter_logits(scores, top_k, top_p)
            next_id = int(torch.multinomial(torch.softmax(scores, -1), 1, generator=generator).item())
        else:
            next_id = forced_id
        ids.append(next_id)
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_id:
            ended_with_eos = forced_id is None
            termination = "eos" if ended_with_eos else "constraint"
            break
    continuation = tokenizer.decode(generated_ids)
    details = {"text": effective_prompt + continuation, "continuation": continuation,
               "unknown_prompt_chars": unknown_prompt_chars, "unknown_prompt_count": unknown_prompt_count,
               "prompt_oov_policy": "unk_preserve_literal",
               "effective_prompt": effective_prompt, "ended_with_eos": ended_with_eos,
               "termination": termination, "generated_tokens": len(generated_ids),
               "ngram_relaxations": relaxations, "hard_form": form}
    if form and (termination != 'constraint' or not form_compliant(details['text'], form)):
        raise RuntimeError('未完成指定诗体，拒绝返回不合规文本')
    return details if return_details else details["text"]


def add_sampling_arguments(parser):
    parser.add_argument("--max-new-tokens", type=int, default=GENERATION_DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=GENERATION_DEFAULTS["temperature"])
    parser.add_argument("--top-k", type=int, default=GENERATION_DEFAULTS["top_k"])
    parser.add_argument("--top-p", type=float, default=GENERATION_DEFAULTS["top_p"])
    parser.add_argument("--repetition-penalty", type=float, default=GENERATION_DEFAULTS["repetition_penalty"])
    parser.add_argument("--no-repeat-ngram-size", type=int, default=GENERATION_DEFAULTS["no_repeat_ngram_size"])


def sampling_config(args):
    return {key: getattr(args, key) for key in ("max_new_tokens", "temperature", "top_k", "top_p",
                                               "repetition_penalty", "no_repeat_ngram_size")}


def resolve_form(value):
    value = DEFAULT_FORM if value is None else value
    if value not in {'wujue', 'qijue', 'free'}:
        raise ValueError('诗体必须为wujue、qijue或free')
    return None if value == 'free' else value


def choose_form(input_fn=None):
    reader = input_fn or input
    choices = {'': DEFAULT_FORM, '1': 'wujue', '五绝': 'wujue', 'wujue': 'wujue',
               '2': 'qijue', '七绝': 'qijue', 'qijue': 'qijue', '3': 'free', 'free': 'free'}
    while True:
        value = reader('选择诗体：1五绝[默认] / 2七绝 / 3自由；q退出：').strip().lower()
        if value == 'q':
            return None
        if value in choices:
            return choices[value]
        print('请输入1、2、3或q。')


def build_parser():
    parser = argparse.ArgumentParser(description='MiniGPT v2：默认E05=1.1，可选整首五绝/七绝')
    parser.add_argument('--checkpoint', type=project_path, default=CHECKPOINT_PATH)
    parser.add_argument('--vocab', type=project_path, default=VOCAB_PATH)
    parser.add_argument('--legacy-model-config', type=project_path, help='仅兼容缺少model_config的历史权重')
    parser.add_argument('--device')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--form', choices=['wujue', 'qijue', 'free'], default=None,
                        help='单次生成默认五绝；交互模式未指定时每首可选')
    add_sampling_arguments(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    legacy = json.loads(args.legacy_model_config.read_text(encoding='utf-8')) if args.legacy_model_config else None
    model, tokenizer = load_model(args.checkpoint, args.vocab, args.device, legacy)
    if args.prompt is not None:
        print(generate(model, tokenizer, args.prompt, seed=args.seed,
                       form=resolve_form(args.form), **sampling_config(args)))
        return
    while True:
        choice = args.form if args.form is not None else choose_form()
        if choice is None:
            break
        prompt = input('请输入诗词开头（回车不指定开头，q退出）：').strip()
        if prompt.lower() == 'q':
            break
        try:
            print(generate(model, tokenizer, prompt, seed=args.seed,
                           form=resolve_form(choice), **sampling_config(args)))
        except ValueError as error:
            print(f'输入不符合所选诗体：{error}')


if __name__ == '__main__':
    main()

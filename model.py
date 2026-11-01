import torch # tensor
import torch.nn as nn # neural network
import torch.nn.functional as F # eg. softmax

import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 最后一维归一化；FP32 计算均方，输出保持输入 dtype，形状不变。
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def make_norm(dim, norm_type):
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"未知 norm_type: {norm_type}")


def apply_rope(x, base=10000.0):
    """x: [B,H,T,D]；相邻两维旋转，位置从 0 开始，无 KV cache。"""
    dim = x.shape[-1]
    if dim % 2 or base <= 0:
        raise ValueError("RoPE 要求偶数 head_dim 和正 base")
    inverse = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angles = torch.outer(torch.arange(x.shape[-2], device=x.device, dtype=torch.float32), inverse)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, position_encoding="learned", rope_base=10000.0):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head # the dimension of token vector
        self.head_dim = n_embd // n_head
        self.position_encoding = position_encoding
        self.rope_base = rope_base
        if position_encoding == "rope" and (self.head_dim % 2 or rope_base <= 0):
            raise ValueError("RoPE 要求偶数 head_dim 和正 rope_base")

        self.qkv = nn.Linear(n_embd, 3*n_embd)
        # X(n_embd) --> Q(n_embd)K(n_embd)V(n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout) # dropout attention matrix
        self.resid_dropout = nn.Dropout(dropout) # dropout output vector

        mask = torch.tril(torch.ones(block_size, block_size)) # mask matrix
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, block_size, block_size)
        )
        # change shape : [batch_size, n_head, q_post, k_posi]
        # register_buffer() : remain a tensor in model

    def forward(self, x): # computation in attention layer
        B, T, C = x.shape
        # [batch_size, token_len, token_dim]
        q, k, v = self.qkv(x).chunk(3, dim = -1)
        # [B, T, 3C] -> 3 * [B, T, C]

        # [B, T, C] ->[B, T, n_head, head_dim] ->[B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.position_encoding == "rope":
            q, k = apply_rope(q, self.rope_base), apply_rope(k, self.rope_base)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim) # scaled dot-product attention

        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v
        # weights: [B, H, T, T]
        # v: [B, H, T, head_dim]
        # out: [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SwiGLU(nn.Module):
    def __init__(self, n_embd, dropout, hidden_dim=None):
        super().__init__()
        # 约 8C/3 的隐藏宽度使三矩阵参数量接近 GELU 的 4C 两矩阵结构。
        hidden_dim = hidden_dim or math.ceil((8 * n_embd / 3) / 8) * 8
        self.gate = nn.Linear(n_embd, hidden_dim)
        self.up = nn.Linear(n_embd, hidden_dim)
        self.down = nn.Linear(hidden_dim, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [B,T,C] -> [B,T,F]；SiLU(xWg) ⊙ (xWu) -> [B,T,C]。
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    # pre-LN transformer block

    def __init__(self, n_embd, n_head, block_size, dropout, norm_type="layernorm",
                 ffn_type="gelu", ffn_hidden_dim=None, position_encoding="learned", rope_base=10000.0):
        super().__init__()

        self.ln1 = make_norm(n_embd, norm_type)
        self.attn = CausalSelfAttention(
            n_embd, n_head, block_size, dropout, position_encoding, rope_base
        )

        self.ln2 = make_norm(n_embd, norm_type)
        if ffn_type not in {"gelu", "swiglu"}:
            raise ValueError(f"未知 ffn_type: {ffn_type}")
        self.ffn = (FeedForward if ffn_type == "gelu" else SwiGLU)(n_embd, dropout, ffn_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

class MiniGPT(nn.Module):
    def __init__(
            self,
            vocab_size,
            block_size=192,
            n_embd=256,
            n_head=4,
            n_layer=4,
            dropout=0.1,
            norm_type="layernorm",
            ffn_type="gelu",
            ffn_hidden_dim=None,
            position_encoding="learned",
            rope_base=10000.0,
    ):
        super().__init__()

        if n_head <= 0 or n_embd <= 0 or n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if block_size < 1 or n_layer < 1 or vocab_size < 1:
            raise ValueError("block_size/n_layer/vocab_size 必须为正数")
        if position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding 必须是 learned 或 rope")
        if ffn_hidden_dim is not None and ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim 必须为正数")

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.position_encoding = position_encoding

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # 保留原拼写：旧 Checkpoint 的 postition_embedding.weight 必须严格匹配。
        self.postition_embedding = nn.Embedding(block_size, n_embd) if position_encoding == "learned" else None
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embd, n_head, block_size, dropout, norm_type, ffn_type,
                ffn_hidden_dim, position_encoding, rope_base
            )
            for _ in range(n_layer)
        ])

        self.ln_f = make_norm(n_embd, norm_type)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias = False)

        self.apply(self._init_weights)

        self.lm_head.weight = self.token_embedding.weight # weight tying

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx:     [B, T]
        targets: [B, T], Padding 标签为 -100

        返回：
        logits: [B, T, vocab_size]
        loss:   标量(training); 未传入 targets 时为 None(Reasoning)
        """
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"the length of the sequence is over block_size({self.block_size})"
            )

        positions = torch.arange(T, device=idx.device)

        tok_emb = self.token_embedding(idx)
        pos_emb = self.postition_embedding(positions) if self.postition_embedding is not None else 0

        x = self.dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None and (targets != -100).any():
            # logits.shape = [B, T, vocab_size]
            # logits.reshape(-1, self.vocab_size) --> logits.shape = [B*T, vocab_size]

            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index = -100,
            )# default : F.cross_entropy(reduction="mean"), average loss
        elif targets is not None:
            # 全 Padding 时定义可微零损失，评估器跳过该 Batch，避免 NaN 污染。
            loss = logits.sum() * 0.0

        return logits, loss

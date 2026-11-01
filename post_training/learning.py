"""Remote-only neural adapters; baseline MiniGPT state_dict stays unchanged."""
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from .prepared import load_prepared, require_coverage
from torch import nn
from torch.nn import functional as F
from checkpoint import load_model_checkpoint, read_checkpoint, save_checkpoint, FORMAT_VERSION
from model import MiniGPT
from tokenizer import CharTokenizer
from .core import (DIMENSIONS, SPLITS, resolve, read_json, read_jsonl, file_hash, digest,
                   new_output, atomic_json, now, source_snapshot, validate_group_splits, validate_sample)


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False


def hidden_states(model, idx):
    if idx.ndim != 2 or idx.shape[1] > model.block_size:
        raise ValueError("Invalid sequence length")
    positions=torch.arange(idx.shape[1],device=idx.device)
    pos=model.postition_embedding(positions) if model.postition_embedding is not None else 0
    return model.ln_f(model.blocks(model.dropout(model.token_embedding(idx)+pos)))


def encode_sequence(sample, tokenizer, block_size):
    text,prompt=sample["text"],sample["prompt"]
    if not text.startswith(prompt):
        raise ValueError("Sample changed literal prefix")
    unknown=sorted({ch for ch in text if ch not in tokenizer.stoi})
    if unknown:
        raise ValueError(f"后训练目标含OOV，不能把UNK当正确汉字训练: {unknown}")
    ids=tokenizer.encode(text)
    if len(ids)-1 > block_size:
        raise ValueError("Training sequence exceeds context window")
    return ids


def supervised_batch(rows, tokenizer, block_size, device):
    sequences=[encode_sequence(row,tokenizer,block_size) for row in rows]
    width=max(len(ids)-1 for ids in sequences)
    x=torch.full((len(rows),width),tokenizer.pad_id,dtype=torch.long,device=device)
    y=torch.full_like(x,-100)
    for index,(row,ids) in enumerate(zip(rows,sequences)):
        length=len(ids)-1
        x[index,:length]=torch.tensor(ids[:-1],device=device)
        targets=ids[1:]
        targets[:len(row["prompt"])]=[-100]*len(row["prompt"])
        y[index,:length]=torch.tensor(targets,device=device)
    return x,y


def full_batch(rows, tokenizer, block_size, device):
    sequences=[encode_sequence(row,tokenizer,block_size) for row in rows]
    if max(map(len,sequences)) > block_size:
        raise ValueError("完整评分序列（含BOS/EOS）超过上下文窗口")
    lengths=torch.tensor([len(ids) for ids in sequences],dtype=torch.long,device=device)
    ids=torch.full((len(rows),int(lengths.max().item())),tokenizer.pad_id,dtype=torch.long,device=device)
    for i,sequence in enumerate(sequences):
        ids[i,:len(sequence)]=torch.tensor(sequence,device=device)
    return ids,lengths


class RewardModel(nn.Module):
    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone=backbone
        self.reward_head=nn.Linear(hidden_size,1)
        nn.init.normal_(self.reward_head.weight,mean=0,std=0.01)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, ids, lengths):
        hidden=hidden_states(self.backbone,ids)
        final=hidden[torch.arange(ids.shape[0],device=ids.device),lengths-1]
        return self.reward_head(final).squeeze(-1)


def ranking_loss(chosen, rejected):
    return F.softplus(rejected-chosen).mean()


def chunks(rows, size):
    for start in range(0,len(rows),size):
        yield rows[start:start+size]


def validate_hyperparameters(config, stage):
    t=config[stage]
    names={"sft":("epochs","batch_size","learning_rate","grad_clip"),
           "reward":("epochs","batch_size","learning_rate","grad_clip"),
           "ppo":("updates","rollout_batch_size","ppo_epochs","learning_rate","value_learning_rate","grad_clip")}[stage]
    for name in names:
        if not isinstance(t[name],(int,float)) or isinstance(t[name],bool) or not math.isfinite(t[name]) or t[name] <= 0:
            raise ValueError(f"{stage}.{name} must be finite positive")
        if name in ("epochs","batch_size","updates","rollout_batch_size","ppo_epochs") and type(t[name]) is not int:
            raise ValueError(f"{stage}.{name} must be integer")
    if t.get("weight_decay",0) < 0:
        raise ValueError("weight_decay must be nonnegative")


def load_policy(checkpoint, vocab, device):
    tokenizer=CharTokenizer.load(vocab)
    model,payload=load_model_checkpoint(checkpoint,tokenizer,torch.device(device),vocab_path=vocab)
    model.eval()
    return model,tokenizer,payload["model_config"]


def export_policy(path, model, model_config, vocab_path, metadata):
    payload={"format_version":FORMAT_VERSION,"model_state_dict":model.state_dict(),
             "model_config":model_config,"data_identity":{"vocab":{"sha256":file_hash(vocab_path),"bytes":Path(vocab_path).stat().st_size}},
             "post_training":metadata}
    save_checkpoint(path,payload)


class TrainingRun:
    def __init__(self, output, stage, config, inputs):
        self.path=new_output(output)
        self.metadata={"stage":stage,"status":"running","started_at":now(),"config":{k:config[k] for k in ("schema_version","seed",stage)},
                       "inputs":{key:file_hash(path) for key,path in inputs.items()},"source":source_snapshot(),
                       "restart_policy":"No exact optimizer resume; use a new run for explicit warm-start."}
        atomic_json(self.path/"run.json",self.metadata)

    def __enter__(self):
        return self

    def log(self, row):
        if any(isinstance(v,float) and not math.isfinite(v) for v in row.values()):
            raise RuntimeError("Nonfinite training metric")
        with (self.path/"metrics.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            handle.flush()

    def __exit__(self, kind, value, traceback):
        self.metadata.update(status="failed" if kind else "completed",finished_at=now())
        if kind:
            self.metadata["error_type"]=kind.__name__
        atomic_json(self.path/"run.json",self.metadata)
        return False


def load_reward(path, tokenizer, vocab_path, device):
    payload=read_checkpoint(path)
    if payload.get("kind") != "post_training_reward_v1" or payload.get("vocab_sha256") != file_hash(vocab_path):
        raise ValueError("Not a compatible v3 reward checkpoint / vocabulary identity mismatch")
    cfg=payload["model_config"]
    if cfg["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("Reward vocabulary size mismatch")
    backbone=MiniGPT(**cfg).to(device)
    backbone.load_state_dict(payload["backbone_state_dict"],strict=True)
    reward=RewardModel(backbone,cfg["n_embd"]).to(device)
    reward.reward_head.load_state_dict(payload["reward_head_state_dict"],strict=True)
    reward.eval()
    for parameter in reward.parameters():
        parameter.requires_grad_(False)
    return reward,payload

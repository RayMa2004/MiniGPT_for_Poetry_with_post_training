"""按实际 optimizer update 计数的调度器，无 PyTorch 依赖。"""
import math


class WarmupCosine:
    def __init__(self, optimizer, base_lr, total_steps, warmup_steps=0, min_lr=0):
        if total_steps < 1 or not 0 <= warmup_steps < total_steps:
            raise ValueError("要求 total_steps > warmup_steps >= 0")
        if not 0 <= min_lr <= base_lr or base_lr <= 0:
            raise ValueError("要求 0 <= min_lr <= base_lr 且 base_lr > 0")
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.completed_steps = 0
        self._apply_next()

    def lr_for_step(self, update):
        """update 从 1 开始；warmup 末步达 base_lr，总预算末步达 min_lr。"""
        if self.warmup_steps and update <= self.warmup_steps:
            return self.base_lr * update / self.warmup_steps
        if self.warmup_steps:
            progress = (update - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        else:
            # 无 warmup 时首个实际更新用 base_lr；单步预算也必须有有效学习率。
            progress = (update - 1) / (self.total_steps - 1) if self.total_steps > 1 else 0.0
        progress = min(1.0, max(0.0, progress))
        return self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2

    def _apply_next(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_for_step(self.completed_steps + 1)

    def step(self):
        self.completed_steps += 1
        self._apply_next()

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ("base_lr", "total_steps", "warmup_steps", "min_lr", "completed_steps")}

    def load_state_dict(self, state):
        for key in ("base_lr", "total_steps", "warmup_steps", "min_lr"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Scheduler 配置不一致: {key}")
        self.completed_steps = state["completed_steps"]
        if not 0 <= self.completed_steps <= self.total_steps:
            raise ValueError("Scheduler 步数超出预算")
        self._apply_next()

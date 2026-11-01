# 奖励模型 best-of-N 与下一轮训练

通用流程见 POST_TRAINING_V3.md。本指南从本包已训练策略出发，使用新输出名称 next1，避免重复运行旧目录。

```bash
python -m post_training sample --tasks post_training_data/fresh1_tasks/tasks.jsonl --checkpoint post_training_runs/round2_ppo/last.pt --samples-per-task 64 --output post_training_data/next1_candidates --device cuda
python -m post_training bestof --samples post_training_data/next1_candidates/samples.jsonl --reward-checkpoint post_training_runs/round2_rm/best.pt --reward-validation post_training_runs/round2_rm/reward_validation.json --config configs/post_training_rm_ppo_experiment.json --output post_training_data/next1_bestof --top-k 8 --batch-size 16 --device cuda
python -m post_training data check --data post_training_data/next1_bestof --section sft
python -m post_training sft --data post_training_data/next1_bestof --init-checkpoint post_training_runs/round2_ppo/last.pt --output post_training_runs/next1_rsft --config configs/post_training_rm_ppo_experiment.json --device cuda
python -m post_training ppo --tasks post_training_data/fresh1_tasks/tasks.jsonl --init-checkpoint post_training_runs/next1_rsft/best.pt --reward-checkpoint post_training_runs/round2_rm/best.pt --reward-validation post_training_runs/round2_rm/reward_validation.json --output post_training_runs/next1_ppo --config configs/post_training_rm_ppo_experiment.json --device cuda
```

这里不调用外部 API。bestof 按任务选候选，去重和划分隔离后数量可能减少；SFT 覆盖检查失败时应扩大候选，不复制 train 到 val。初始策略必须与采样权重哈希相同。旧奖励模型可能偏离新策略分布，下一轮应按通用指南重新生成偏好并训练新奖励模型。

请用相同任务与采样索引独立比较 next1_rsft 与 next1_ppo，开发用 val，最终才查看 test。既有 first/round2 目录属于训练来源和有效资产，不要覆盖。

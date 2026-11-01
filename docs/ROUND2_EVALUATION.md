# 第二轮评估

第二轮训练已完成，不需要重复训练。默认生成使用第二轮 PPO；原预训练基线、首轮 PPO、第二轮 RSFT/PPO 均保留。

```bash
python scripts/round2_eval.py --split val
python scripts/round2_eval.py --split test
```

这两条仅检查输入和显示计划，不读取密钥、不创建输出、不请求 API。输入固定为 fresh1_candidates、ppo_v1_candidates、round2_rsft_eval、round2_ppo_eval 的 samples.jsonl，已随包保留。

`post_training_data/round2_full_eval_val/` 保留已付费的成功标签和对应输入，可复用。显式添加 `--execute` 才会进入标注/汇总阶段，需要 POSTTRAIN_API_KEY；程序会检查缓存的内容哈希与评分器身份，仅请求未完成项。即使缓存完整，目前执行入口也要求已设置密钥，因此不要把 --execute 当作无网络的纯离线汇总命令。改变评分服务配置可能使旧缓存身份不匹配，应使用新的 --output 目录。

```bash
python scripts/round2_eval.py --split val --execute
```

执行前核对 API 计划和费用；这条命令不是本次验收的一部分。重复执行可恢复标注，不会为已成功且身份匹配的项目重复收费。test 计划只用于最终方案比较，不要据 test 结果继续反复调参。

汇总包括配对偏好、四维评分和阈值合格率。PPO 文件名 best.pt 是默认入口兼容名称，不证明诗歌质量最优。AI 偏好与训练奖励都不等于独立人工评价。

# AI 反馈后训练操作指南

## 工作方式

主干与词表不变，按任务生成候选，再由评分或偏好标签准备训练数据，执行 RSFT、奖励模型和 PPO。输出必须位于本项目 post_training_data/ 或 post_training_runs/ 的新子目录，不能跨目录混用权重、词表、标签和验证报告。

下列命令是从当前默认策略开始的新一轮示例，不会重复覆盖随包运行。使用时将 `release1` 换成新名称。GPU 推荐 cuda；CPU 只适合校验和小规模实验。训练、采样中断后使用新输出目录；AI 标注则重用同一标签路径以恢复进度。

## 1. 配置评分服务

编辑 configs/post_training_v3.json 中 api.base_url、api.model 等字段。历史配置不代表服务仍可用。密钥通过当前终端的 POSTTRAIN_API_KEY 环境变量传入，不保存到项目。

PowerShell 可在终端自行设置 `$env:POSTTRAIN_API_KEY`；Linux 用 `export POSTTRAIN_API_KEY=...`。不要把真实值写入脚本、文档或日志。以下 judge 命令省略 --execute 时只显示计划；添加后将产生外部 API 请求及费用，执行前先核对计划。遇到 request 上限，重复同一命令，直到 remaining_items 为 0。

## 2. 新任务与候选

```bash
python -m post_training data tasks --prefixes post_training/prompts.json --namespace release1 --output post_training_data/release1_tasks
python -m post_training sample --tasks post_training_data/release1_tasks/tasks.jsonl --checkpoint checkpoints/best.pt --samples-per-task 8 --output post_training_data/release1_candidates --device cuda
python -m post_training judge score --input post_training_data/release1_candidates/samples.jsonl --output post_training_data/release1_candidates/scores.jsonl
```

核对后给最后一条添加 --execute。候选内记录 checkpoint/vocab 的哈希、任务组、划分、采样参数和种子。不要把训练文本复制进 val/test；已选定开发数据后，test 应保留到最终比较。

## 3. 准备与训练 RSFT

标注完整后：

```bash
python -m post_training data prepare --samples post_training_data/release1_candidates/samples.jsonl --scores post_training_data/release1_candidates/scores.jsonl --output post_training_data/release1_sft_data
python -m post_training data check --data post_training_data/release1_sft_data --section sft
python -m post_training sft --data post_training_data/release1_sft_data --init-checkpoint checkpoints/best.pt --output post_training_runs/release1_rsft --device cuda
```

数据检查必须通过：train/val 各覆盖给定开头/空开头 × 五绝/七绝四类。若筛后覆盖不足，增加/校准候选并准备新数据集。SFT 初始化必须与生成候选的权重一致；不要关闭哈希校验。best.pt 依据验证续写 loss 选择，仅说明训练选择标准。

## 4. 准备偏好与训练奖励模型

```bash
python -m post_training data pairs --samples post_training_data/release1_candidates/samples.jsonl --max-per-task 4 --output post_training_data/release1_candidates/pairs.jsonl
python -m post_training judge pairs --input post_training_data/release1_candidates/pairs.jsonl --output post_training_data/release1_candidates/preferences.jsonl
```

计划核对后给 judge 添加 --execute，重复至全部成功；然后：

```bash
python -m post_training data prepare --samples post_training_data/release1_candidates/samples.jsonl --pairs post_training_data/release1_candidates/pairs.jsonl --preferences post_training_data/release1_candidates/preferences.jsonl --output post_training_data/release1_rm_data
python -m post_training data check --data post_training_data/release1_rm_data --section preferences
python -m post_training reward --data post_training_data/release1_rm_data --init-checkpoint post_training_runs/release1_rsft/best.pt --output post_training_runs/release1_rm --device cuda
```

奖励模型是主干加评分头，不可传给 generate.py。只有配套 reward_validation.json 的 passed=true、权重/词表/数据身份及准入门槛全部匹配时，才能进入 PPO。失败应收集更可靠偏好，不应为了放行而事后降低阈值。

## 5. PPO

```bash
python -m post_training ppo --tasks post_training_data/release1_tasks/tasks.jsonl --init-checkpoint post_training_runs/release1_rsft/best.pt --reward-checkpoint post_training_runs/release1_rm/best.pt --reward-validation post_training_runs/release1_rm/reward_validation.json --output post_training_runs/release1_ppo --device cuda
python generate.py --checkpoint post_training_runs/release1_ppo/last.pt --form wujue --prompt '春风' --device cuda
```

PPO 用独立的冻结参考策略、KL 约束和价值头；导出的策略与原 MiniGPT 生成格式兼容。last.pt 是最终导出，并非独立验证最优。trainer_state.pt 仅为诊断，源码不支持精确优化器恢复；失败后新建运行显式热启动。SFT/RM 同样使用新的输出目录，不提供历史运行的原地续训命令。

## 6. 生成与匹配比较

```bash
python -m post_training sample --tasks post_training_data/release1_tasks/tasks.jsonl --checkpoint post_training_runs/release1_rsft/best.pt --samples-per-task 8 --output post_training_data/release1_rsft_eval --device cuda
python -m post_training sample --tasks post_training_data/release1_tasks/tasks.jsonl --checkpoint post_training_runs/release1_ppo/last.pt --samples-per-task 8 --output post_training_data/release1_ppo_eval --device cuda
python -m post_training compare build --baseline post_training_data/release1_rsft_eval/samples.jsonl --candidate post_training_data/release1_ppo_eval/samples.jsonl --split val --output post_training_data/release1_comparison
python -m post_training judge pairs --input post_training_data/release1_comparison/pairs.jsonl --output post_training_data/release1_comparison/preferences.jsonl
```

若 API pairs>0，确认计划后给 judge 添加 --execute。完成后运行：

```bash
python -m post_training compare report --comparison post_training_data/release1_comparison/comparison.json --preferences post_training_data/release1_comparison/preferences.jsonl --output post_training_data/release1_report
```

若所有输出一致、pairs=0，跳过 judge 且 report 省略 --preferences。比较要固定任务、采样设置和索引。AI 盲评反映该评分器偏好，不等于人工质量结论。

## 本包状态

默认 checkpoints/best.pt 是第二轮 PPO，原预训练权重在 checkpoints/pre_posttraining_baseline.pt，原 provenance 另保留。初代/第二轮 RSFT 与 PPO、初代/第二轮有效 RM 均保留；第二轮策略对应 post_training_runs/round2_ppo/last.pt。

可直接检查 post_training_data/round2_bestof 的 SFT 与 round2_rm_prepared 的偏好。round2_candidates 与 first/round2 的评分、偏好及候选保留，用于重建训练输入。首轮 RM 历史门槛为 0.64，第二轮 RM 也使用实验配置；不可据此声称已达到独立人工质量要求。

[第二轮评估指南](ROUND2_EVALUATION.md)说明如何复用已付费标签；[best-of-N 指南](POST_TRAINING_V4.md)说明下一轮采样与训练。

# mini_gpt_v4.2 操作指南

在项目根目录执行，依赖安装与生成命令见 README。CPU 验证把 `--device cuda` 换成 `--device cpu`；完整配置的训练仍建议 GPU。

## 预训练

```bash
python train.py --experiment-id release_train --seed 42 --device cuda
```

输出 `runs/release_train_seed42/`，已有运行 ID 拒绝覆盖。若要从发布权重继续学习，创建新运行并加 `--init-checkpoint checkpoints/best.pt`；这会重置优化器与训练状态，属于热启动。

精确恢复本版本源码创建的运行：

```bash
python train.py --resume runs/release_train_seed42/last.pt --device cuda
```

不要同时附加新配置或新 seed。历史发布权重与当前源码指纹可能不同，不放宽校验来冒充精确恢复。训练不会自动替换发行目录 `checkpoints/` 中的权重。

## 从 raw 创建新数据版本

原始输入随包位于 `data/raw/`，共 58 个分片。使用原清洗规则创建独立目录：

```bash
python prepare_data.py --raw-dir data/raw --output-dir data_rebuilt --rules configs/data_cleaning_v2.json
```

工具拒绝覆盖当前 data、checkpoints 及已存在输出。新数据会有新词表和新指纹，需显式新配置、新训练，不应替换当前词表后继续加载旧权重。v1 原数据使用较早清洗流程，上述命令采用 v2 保守规则，不承诺与 v1 原 train/val 字节相同。v2 后续的 raw 已按历史 manifest 验证，但环境或 OpenCC 版本变化也可能影响重建字节。

## 评估

```bash
python evaluate.py --evaluation-id release_eval --device cuda --samples-per-prompt 4 --hard-form
```

评估输出新目录；重复 ID 拒绝覆盖。`--hard-form` 只对提示文件中标注诗体的任务生效。保留未标注诗体的自由生成诊断。历史训练 loss 和本轮生成质量是不同证据。

## 校验与问题处理

```bash
python scripts/static_audit.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

出现词表 SHA256 或 vocab_size 不匹配时，恢复本版本词表，不能通过删校验或只比较大小解决。已有输出目录冲突时使用新 ID。源代码指纹不匹配的历史训练请选择热启动新运行。无 GPU 的机器显式选择 cpu；Windows 测试已显式使用 UTF-8，保存权重使用可写句柄 fsync。

## 文件职责

`model.py` 为模型，`generate.py` 为用户生成入口；`train.py`、`checkpoint.py`、`experiment.py`、`scheduler.py` 管理训练及恢复；`prepare_data.py`、`data_cleaning.py`、`tokenizer.py`、`dataset.py` 处理数据。`evaluation/` 与 `evaluate.py` 提供评估；`configs/` 为参数配置，`tests/` 为回归测试。`RELEASE_MANIFEST.json` 绑定随包源码、文档、数据和权重。

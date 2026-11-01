# MiniGPT v4.2

完成第二轮 RSFT/PPO，并提供第二轮配对及绝对质量评估工具。延续 v4 的 best-of-N、奖励模型与完整后训练链。

这是 GPT 风格的微型解码器教学项目，不是 GPT-3 原版模型、官方权重或通用聊天模型。

## 当前随包资产

默认 checkpoints/best.pt 是第二轮 PPO 的原样副本（文件名不代表独立质量最优）。pre_posttraining_baseline.pt 保留原预训练基线；另保留首轮/第二轮 RSFT、PPO 与奖励模型。

保留 raw、预训练数据、两轮任务/候选/标注/准备数据及已付费评估标签，可复用离线缓存。第二轮 PPO 尚无独立人工质量优于 RSFT 的证据。

完整文件和 SHA256 见 `RELEASE_MANIFEST.json`。本项目可独立复制和解压使用，不依赖相邻版本目录。

## 安装

Python 3.12 或以上。在项目根目录执行以下命令：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS 改用：source .venv/bin/activate
python -m pip install -r requirements.txt
```

也可用 `uv sync`。GPU 训练请先安装适合本机驱动的 CUDA 版 PyTorch，再安装其余依赖；用 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 核对。CPU 可用于验证和小规模生成，完整训练建议 CUDA GPU。无需历史的 `myconda` 环境或 `/mnt/...` 绝对路径。

## 生成

```bash
python generate.py --prompt '春风' --form wujue --device cuda --seed 42
python generate.py --prompt '江南' --form qijue --device cuda --seed 42
python generate.py --device cuda
```

最后一条进入交互模式。无 GPU 时将 `cuda` 换成 `cpu`。五绝四句每句 5 字，七绝四句每句 7 字，标点为“，。，。”；格式约束不保证押韵、平仄或语义。显式 --form free 进入自由模式；词表外输入按 UNK 编码并保留原字显示，不能证明模型理解未知字。CLI 相对路径按项目根目录解析。

## 训练、数据处理与验证

见 [操作指南](docs/OPERATIONS.md)；[模型说明](docs/MODEL_CARD.md)记录模型用途、权重角色和限制。先执行：

```bash
python scripts/static_audit.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

`static_audit.py` 校验发行时文件的 SHA256、Python 语法和 JSON/TOML，不调用模型或网络。修改项目后原始发行校验会报告差异，这是完整性检查的预期行为。

## 后训练

详见 [后训练操作指南](docs/POST_TRAINING_V3.md)，其中注明本版本现有成果及可直接使用的数据。仅 `judge ... --execute` 或显式执行评估才会请求第三方 API。密钥通过 `POSTTRAIN_API_KEY` 环境变量设置，不写入配置文件。

第二轮缓存评估见 [第二轮评估指南](docs/ROUND2_EVALUATION.md)。

## 发行信息

本次移出了本地环境、Python/Notebook 缓存、历史实验输出、错误日志和已淘汰的实验权重；保留训练/生成实现、回归测试、有效数据及阶段成果。历史路径和源码指纹在 provenance/run.json 中是历史事实，不改写为本次发行路径。预训练权重可显式热启动新运行；严格续训只适用于当前源码创建的运行。

验证范围见 [验收记录](docs/VERIFICATION.md)。数据与第三方依赖说明见 [来源及授权](THIRD_PARTY_NOTICES.md)。

许可：保留全部权利，仅供获准接收者学习、研究和本地运行；公开再分发、商业使用需另行授权。详见 [LICENSE](LICENSE)。

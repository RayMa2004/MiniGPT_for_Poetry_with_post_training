# 验收记录（2026-10-31）

环境：Windows、Python 3.13.5、PyTorch 2.14.1+cpu。所有计算在 CPU；未请求真实评分 API。

## 回归测试

完整套件 118 项，除 1 项需指定真实权重的条件测试外均通过；随后设置本包权重/词表路径单独执行该项也通过（共 118 项全部实际覆盖）。

测试覆盖模型、保存/加载、训练恢复、随机状态、生成、调度、数据及本版本具有的后训练功能。API 测试使用离线替身；不代表外部服务已连通。

## 随包权重实测

| 路径 | 角色 | 结果 |
|---|---|---|
| `checkpoints/best.pt` | policy | 加载、生成及诗体检查通过 |
| `checkpoints/pre_posttraining_baseline.pt` | policy | 加载、生成及诗体检查通过 |
| `post_training_runs/fresh1_rsft/best.pt` | policy | 加载、生成及诗体检查通过 |
| `post_training_runs/ppo_v1/last.pt` | policy | 加载、生成及诗体检查通过 |
| `post_training_runs/rm_ppo_experiment/best.pt` | reward | 奖励计算有限，身份门槛通过 |
| `post_training_runs/round2_ppo/last.pt` | policy | 加载、生成及诗体检查通过 |
| `post_training_runs/round2_rm/best.pt` | reward | 奖励计算有限，身份门槛通过 |
| `post_training_runs/round2_rsft/best.pt` | policy | 加载、生成及诗体检查通过 |

每个策略均实际加载，GPU 版本分别生成五绝/七绝并验证结构、执行短自由生成。local 进行自由续写检查。默认 best.pt 另执行有限 loss、反向传播、梯度有限性与一次优化器更新，仅修改内存模型，未回写发布权重。

## 数据与训练链

58 个 raw 分片全部匹配历史清单的 SHA256。v2 已完整执行保守数据重建，train.txt/val.txt 逐字节一致；词表 ID 完全一致，Windows CRLF 与原 LF 导致词表文件哈希不同。v2–v4.2 的预处理、清洗、分词器源码相同。本包原词表字节保持不变。

## 后训练输入检查

所有保留的 prepared 数据经过内容哈希、划分及样本结构检查。各 section 的训练覆盖状态：

- `fresh1_prepared/sft`：就绪，train=58、val=10。
- `fresh1_prepared/preferences`：覆盖不足，训练器会拒绝；对应其他 section 仍可使用，train=0、val=0。
- `pilot_prepared/sft`：覆盖不足，训练器会拒绝；对应其他 section 仍可使用，train=4、val=0。
- `pilot_prepared/preferences`：就绪，train=124、val=16。
- `rm_v1_prepared/sft`：覆盖不足，训练器会拒绝；对应其他 section 仍可使用，train=0、val=0。
- `rm_v1_prepared/preferences`：就绪，train=1529、val=191。
- `round2_bestof/sft`：就绪，train=768、val=96。
- `round2_bestof/preferences`：覆盖不足，训练器会拒绝；对应其他 section 仍可使用，train=0、val=0。
- `round2_rm_prepared/sft`：覆盖不足，训练器会拒绝；对应其他 section 仍可使用，train=0、val=0。
- `round2_rm_prepared/preferences`：就绪，train=1535、val=192。

SFT 数据集中的 preferences 为空、偏好数据集中的 sft 为空是不同用途，并非清理丢失数据。pilot 的 SFT 验证集为空是历史校准状态，不能直接用于完整训练。

## 发行边界

未执行完整 GPU 预训练或后训练、未验证 CUDA 数值/吞吐、未支付 AI 评分费用、未声明诗歌质量优于前一版本。训练恢复测试使用小夹具，不能替代长期 GPU 任务验收。

所有保留的原数据、词表和原始权重均按整理前 SHA256 核对；v1 combined_best.pt 是原选定权重的逐字节副本。原始历史 provenance/run.json 不改写。发行文件清单及静态校验工具随包提供。

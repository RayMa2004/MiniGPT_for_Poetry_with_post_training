# 数据与第三方组件

## 数据来源

随包 `data/raw/poet.tang.*.json` 来自 chinese-poetry/chinese-poetry 的全唐诗分片。来源仓库：https://github.com/chinese-poetry/chinese-poetry ，分片目录为“全唐诗”。本次未下载替换语料；从已有本地副本保留或恢复，内容哈希见 RELEASE_MANIFEST.json。v2 及后续版本的恢复还与 data/manifest.json 中的 raw_files 逐个匹配。

诗歌原文、数据整理成果、元数据及 API 生成标签可能适用不同权利或服务条款。本包未附来源仓库完整许可文本，未声明它们与本项目代码自动适用相同授权；公开再分发前应确认来源仓库与所用评分服务的适用条款。

## 软件依赖

PyTorch、NumPy、OpenCC、TensorBoard 及可选 Matplotlib 由用户通过包管理器安装，各自保留其原有授权。本包不捆绑机器上的虚拟环境或第三方二进制库。

## 项目代码

作者已选择“保留全部权利，仅分享使用”，具体范围见 LICENSE。该许可适用于权利人有权授权的代码、文档和模型成果，不替代第三方数据及服务条款。

## 配置与凭据

API 地址和模型名是历史配置，运行前按实际服务修改。评分密钥仅从 POSTTRAIN_API_KEY 环境变量读取。本次验收不向外部评分服务发送数据。

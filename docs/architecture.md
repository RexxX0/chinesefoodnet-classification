# 架构审查与整理记录

## 文件职责

| 模块 | 职责 |
| --- | --- |
| food-classifier/src/train.py | 训练、恢复、日志与轮换验证 |
| dataset.py | ImageFolder、数据变换、样本采集与分折 |
| infer.py | 训练权重加载、标签和单图 CLI |
| evaluate.py | 复用训练端 evaluate 计算验证指标 |
| app.py | 复用 infer 的 Gradio 演示 |
| config.py | 统一配置读取和路径解析 |
| food-web/server.py | HTTP 路由、生命周期、静态文件 |
| food-web/inference.py | Web 推理和不同检查点格式兼容 |
| food-web/static/ | 页面、样式与同源 API 请求 |
| 各 scripts/ | 人工或部署调用入口，不能依据没有 import 判定为无用 |

当前规模适合两个脚本式子项目，不需要增加复杂安装包或服务层。未来确有第三个推理消费者时可再提取共享包。

## 已整理

- 根目录增加 Git 忽略规则、换行约定、编辑器约定、Ruff 和 pytest 配置。
- CI 从 food-web/.github/workflows/ 迁到根目录，覆盖两个模块。
- 修复所有默认入口引用不存在 EVA02 配置的问题；Gradio 支持 --config。
- 提取统一配置读取器，YAML 内路径相对于配置文件。
- 新增完整 208 类配置；原 32 类实验设置保留，机器专用路径改为相对路径。
- 两份标签 SHA-256 一致，合并到 data/metadata/class_names.csv；训练端读取支持 BOM。
- logs/history.csv 移到 reports/experiments/legacy/；两张图片移到 reports/figures/。
- 训练和 Web 默认共享 outputs/convnextv2_base/best.pth。
- Docker 改为根目录构建上下文，以复制共享类别字典。

## 清理依据

删除的唯一数据副本为 food-web/class_names.csv，删除前比对了哈希。历史训练记录、TensorBoard 事件和 cloudflared 均未删除；后两类仅被 Git 忽略。旧空目录可能仍留在本地，Git 不追踪空目录。

未合并两套权重加载器：训练端严格载入，Web 端支持更多封装并宽松匹配，直接合并会改变行为。Gradio 与 FastAPI 有不同用途，均保留。

## 后续改进

- 轮换验证样本会在其他阶段参与训练，不是标准独立交叉验证。默认配置已关闭它。
- Web 允许随机/部分权重运行，可在后续单独变更为严格加载和启动失败。
- 检查点未保存完整配置、类别映射和依赖版本，需补齐可复现信息。
- set_seed 主要设置 torch 种子，并未提供完整确定性训练保证。
- 当前依赖没有全量锁定，应从实际验证过的训练环境导出版本。
- 没有独立 test split CLI、混淆矩阵或逐类指标，本次未伪造这些能力。

参考：[PyTorch ImageNet 示例](https://github.com/pytorch/examples/tree/main/imagenet) 与 [Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science)，采用清晰入口以及代码/数据/报告分离原则。

## 本次验证记录

本地 Python 3.13.5 环境：Ruff 检查通过；7 项 pytest 回归检查通过；源码编译检查通过；train/evaluate/infer/app 的 --help 均成功；文档链接、UTF-8 编码、YAML 与忽略规则核对通过。

测试覆盖配置在不同工作目录下的路径解析、绝对/相对/空路径、208 类标签一致性、BOM 首行读取、Web 生命周期、静态资源、图片推理接口和错误上传。Web 使用小型假模型，不下载权重。

训练端 TensorBoard 导入延后到实际训练入口，评估及 --help 不再要求先加载日志依赖。本机缺少 tensorboard，完整训练仍需安装 requirements.txt。没有数据集或真实权重，未执行训练、真实准确率评估或 Docker 构建；GitHub 上的 Python 3.10–3.12 矩阵尚待推送后运行。

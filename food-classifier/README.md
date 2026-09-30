# 训练与离线推理

负责训练、验证、命令行推理和 Gradio 演示。以下命令从仓库根目录执行。

## 安装与数据

```bash
python -m pip install -r food-classifier/requirements.txt
python food-classifier/scripts/check_cuda.py
```

GPU 环境需要安装与驱动匹配的 PyTorch。数据集来源：[ChineseFoodNet 官方网站](https://sites.google.com/view/chinesefoodnet/)。下载后整理为 ImageFolder 格式，详见 [数据说明](../data/README.md)。

## 配置

| 文件 | 用途 |
| --- | --- |
| configs/convnextv2_base.yaml | 默认 208 类、全部样本、固定验证集 |
| configs/convnextv2_test.yaml | 原有 32 类 / 每类最多 120 张 / 轮换验证实验设置 |
| configs/*.local.yaml | 个人完整配置副本，不提交 |

配置不支持继承。YAML 内相对路径以 YAML 所在目录为基准；CLI 的 --config、--image、--resume 参数相对于当前工作目录。绝对路径可直接使用。

默认 model.pretrained 为 true，由 timm 下载骨干权重。离线时先准备对应文件，设置 pretrained 为 false 和 pretrained_checkpoint。骨干预训练权重不能直接作为已训练的 208 类分类权重使用。

## 命令

```bash
python food-classifier/src/train.py
python food-classifier/src/train.py --config food-classifier/configs/convnextv2_test.yaml
python food-classifier/src/train.py --resume outputs/convnextv2_base/latest_checkpoint.pth
python food-classifier/src/evaluate.py
python food-classifier/src/infer.py --image path/to/food.jpg --top-k 5
python food-classifier/src/app.py --config food-classifier/configs/convnextv2_base.yaml
```

`evaluate.py` 评估 paths.val_dir；test_dir 目前没有独立入口。评估自行准备的有标签独立测试集时，可创建配置副本，将 val_dir 指向符合 ImageFolder 布局的测试集，并关闭样本限制。官网提供的测试集没有真实标签，不能直接用此方法计算准确率。

默认 auto_resume 为 true；重新运行会接着最新检查点训练。新实验应使用新的空输出目录与对应检查点路径。

## 输出与脚本

默认输出到仓库的 outputs/convnextv2_base/：best.pth（最佳权重）、latest_checkpoint.pth（恢复训练）、logs/history.csv、logs/tensorboard/、metrics.png。

```bash
bash food-classifier/scripts/train.sh
bash food-classifier/scripts/serve.sh
bash food-classifier/scripts/tensorboard.sh
bash food-classifier/scripts/train_cloud.sh configs/convnextv2_base.yaml
```

shell 启动器先切换到 food-classifier/，传入脚本的相对路径以该目录为基准。

## 评估约定

原有 cross_validation.enabled 会合并 train/val，在同一个持续训练的模型上轮换验证折。这不是每折重新初始化的标准交叉验证，后续验证样本可能已经参与训练。保留该实现供复现实验；默认完整配置关闭它，不能把轮换验证结果当成独立测试性能。

## Docker

构建上下文必须为仓库根目录，以包含共享字典：

```bash
docker build -f food-classifier/Dockerfile -t chinesefoodnet-classifier .
docker run --rm -p 7860:7860 --mount type=bind,source=/absolute/path/to/outputs,target=/app/outputs chinesefoodnet-classifier
```

运行前宿主机 outputs/convnextv2_base/best.pth 必须存在。镜像默认启动 Gradio，数据与权重不打包进镜像。

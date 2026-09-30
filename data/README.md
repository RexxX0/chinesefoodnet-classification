# 数据目录

## 数据集来源

本项目使用 [ChineseFoodNet 官方网站](https://sites.google.com/view/chinesefoodnet/) 提供的数据集。数据集介绍、获取方式及使用要求以官网为准。本仓库不包含数据集图片，也不提供自动下载脚本。

实际使用的数据版本、样本数量和 train/val/test 划分仍需结合本项目的训练记录补充，不应将官网信息直接视为本项目实际使用的划分。

官网公布的数据包含 208 类：训练集 145,065 张、验证集 20,253 张、测试集 20,310 张。train 和 val 按 `000`–`207` 分类目录组织；官方测试集只提供图片，不提供真实标签，因此不能直接用本项目的 ImageFolder 评估脚本计算该测试集准确率。下面的 `test/000/` 仅表示自行准备的**有标签测试集**布局，不代表官方下载包的测试目录。

数据集论文：[ChineseFoodNet: A Large-scale Image Dataset for Chinese Food Recognition](https://arxiv.org/abs/1705.02743)，引用格式可参见官网的 CITE 部分。

## 本地组织

仅 metadata/class_names.csv 随代码提交。原始数据放 raw/，整理后的数据放 processed/，两者已忽略。

```text
data/
├── metadata/class_names.csv
└── processed/chinesefoodnet/
    ├── train/000/*.jpg
    ├── train/001/*.jpg
    ├── ...
    ├── val/000/*.jpg
    └── test/000/*.jpg
```

每个 split 应包含相同类别目录。ImageFolder 按目录名字典序编号，推荐 000 到 207 这样的零填充编号，避免无填充数字被排序为 0, 1, 10, 100, ...。

类别字典为 UTF-8 CSV，列为 id,zh_name,en_name，ID 为 0–207。模型输出索引必须对应 CSV 的 ID。已有数据顺序不同需重新对齐，不能只改类别数量。

数据下载与 train/val/test 划分不在当前代码中实现。请记录来源、版本、每类数量和划分方案，保持测试集不参与训练。其他布局可在个人配置中使用绝对路径。

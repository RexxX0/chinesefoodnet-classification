# 模型与实验说明

| 项目 | 当前信息 |
| --- | --- |
| 任务 | 208 类中餐图像分类 |
| 骨干 | convnextv2_base.fcmae_ft_in22k_in1k |
| 框架 | PyTorch / timm |
| 输入 | RGB，224 × 224 |
| 验证预处理 | Resize(int(224 × 1.14)) → CenterCrop → ImageNet 归一化 |
| 训练增强 | RandomResizedCrop、RandomHorizontalFlip |
| 损失 / 优化器 | CrossEntropyLoss / AdamW |
| 数据来源 | [ChineseFoodNet 官方网站](https://sites.google.com/view/chinesefoodnet/) |
| 实际使用的数据版本、样本数、划分 | 待作者补充 |
| 官网公布的划分 | train 145,065 / val 20,253 / test 20,310 张；官方 test 不提供真实标签 |
| 训练硬件、精确依赖版本 | 待作者补充 |
| 完整 208 类独立测试成绩 | 尚未提供可核验结果 |
| 权重下载地址与校验值 | 尚未发布 |

## 已有资料

- [历史 CSV](../reports/experiments/legacy/history.csv)
- [原 logs/metrics.png](../reports/figures/legacy_metrics.png)
- [原 training_curves.png](../reports/figures/training_curves.png)

以上资料原样保留，缺少运行元数据证明它们属于同一实验。原配置使用 32 类、每类最多 120 张、10 epoch 和轮换验证，不能直接将这些材料表述为完整 208 类基准结果。

新增完整配置尚未重新训练。evaluate.py 输出指定 val_dir 上的 loss 和 top-1 accuracy，并非自动评估独立测试集。

## 限制

模型为固定类别分类器。非食物和字典外菜品仍可能被分到已知类别；softmax 分数不等于经过校准的可靠性保证。模型不能据此判定食物安全、过敏原或营养成分。

复现实验应同时保存完整配置、数据划分、类别映射、软件版本、硬件、权重校验值和独立测试指标。

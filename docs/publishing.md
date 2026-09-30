# 上传 GitHub

建议仓库名为 chinesefoodnet-classification；本地 Food_Clacification 名称不影响运行，无需先改名。

## 发布准备

1. 核对已添加的 ChineseFoodNet 官网来源，补充作者信息、实际数据版本和实验结果。
2. 根据自己的授权意愿选择许可证并添加 LICENSE，本次没有代选。
3. 安装 requirements-dev.txt 后运行 `python -m ruff check .` 和 `python -m pytest -q`。
4. 数据、权重、下载程序留在本机。使用 Git 提交；网页拖拽上传不会按本地 .gitignore 过滤整个文件夹。

## 初始化与推送

首次建立仓库时，在仓库根目录执行（已初始化的仓库跳过 git init）：

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare ChineseFoodNet training and web demo repository"
```

在 GitHub 创建空仓库，替换下面 URL：

```bash
git remote add origin https://github.com/YOUR_USERNAME/chinesefoodnet-classification.git
git push -u origin main
```

暂存清单应有源码、配置、文档、测试、字典和报告，不应有模型权重、数据集图片、TensorBoard 日志或 cloudflared.exe。忽略规则不会取消已有文件的跟踪；在其他现有仓库中使用时也需检查暂存清单。

根目录工作流会在推送或 PR 时运行。真实权重可单独发布并说明地址、模型配置和类别顺序。

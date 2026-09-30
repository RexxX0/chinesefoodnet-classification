# ChineseFoodNet Web

FastAPI 后端与原生 HTML/CSS/JavaScript 前端，提供图片上传、Top-K 分类和中英文标签展示。

## 启动

从仓库根目录运行：

```bash
python -m pip install -r food-web/requirements.txt
python -m uvicorn server:app --app-dir food-web --host 127.0.0.1 --port 8000
```

默认读取仓库内 outputs/convnextv2_base/best.pth 和 data/metadata/class_names.csv；默认路径不依赖启动目录。也可进入 food-web/，使用 scripts/run_dev.bat（Windows）或 bash scripts/run_dev.sh（Linux/macOS）。

界面：<http://127.0.0.1:8000>；接口文档：<http://127.0.0.1:8000/docs>。

## 环境变量

| 变量 | 默认值 / 用途 |
| --- | --- |
| CKPT_PATH | 仓库的 outputs/convnextv2_base/best.pth |
| CLASS_CSV | 仓库的 data/metadata/class_names.csv |
| MODEL_NAME | convnextv2_base.fcmae_ft_in22k_in1k |
| IMG_SIZE | 224 |
| NUM_CLASSES | 208 |
| TOP_K | 5 |
| DEVICE | 自动 CUDA / CPU |
| MAX_UPLOAD_MB | 10 |
| CORS_ORIGINS | *，逗号分隔 |

自定义路径建议使用绝对路径；环境变量中的相对路径以启动目录为基准。PowerShell 示例：

```powershell
$env:CKPT_PATH = 'D:/models/best.pth'
python -m uvicorn server:app --app-dir food-web --host 127.0.0.1 --port 8000
```

Linux/macOS 可在命令前加 CKPT_PATH=/path/to/best.pth。

## 接口

| 方法与路径 | 用途 |
| --- | --- |
| GET /api/health | 模型、设备、权重信息与警告 |
| GET /api/classes | 类别 ID 与中英文名称 |
| POST /api/predict?top_k=5 | multipart 字段 file 上传图片 |

```bash
curl -F "file=@food.jpg" "http://127.0.0.1:8000/api/predict?top_k=5"
```

## 运行边界与检查

当前加载器保留原有宽松行为：缺失权重时用随机参数，形状不匹配时只加载匹配项。请检查健康接口的 loaded_tensors 和 head_warning；服务能启动不等于模型可用。真实预测需要结构、类别顺序和尺寸一致的完整训练权重。

当前定位为演示应用。每个 worker 均会持有模型，部署时需考虑内存；公开运行前按环境配置访问控制和 CORS。临时演示脚本见 [TUNNEL.md](TUNNEL.md)，bin/ 下载文件不提交。

根目录 .github/workflows/ci.yml 统一检查。tests/test_web.py 使用小型假模型检查生命周期、静态资源和图片上传，不替代真实权重验证。

# YOLO11 Learning

这是我的第一个视觉任务复现项目。

## 项目目标

使用 Ultralytics YOLO11 在 AutoDL RTX4090 服务器上完成图片目标检测推理。

## 工程规范

- 代码目录：/root/autodl-tmp/projects/yolo11-learning
- Conda 环境：yolo
- 模型权重：yolo11n.pt
- 输入图片：bus.jpg
- 输出结果：outputs/ 或 runs/

## 最小运行命令

```bash
conda activate yolo
python scripts/yolo_infer.py
cat > scripts/yolo_infer.py <<'EOF'
from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO("yolo11n.pt")

# 2. 对图片做推理
results = model("https://ultralytics.com/images/bus.jpg", save=True)

# 3. 打印结果
print("推理完成")
print(results[0])
## Environment

### Create environment

```bash
conda env create -f environment.yml
```

### Activate environment

```bash
conda activate yolo
```

### Verify installation

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```
from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO("yolo11n.pt")

# 2. 对图片做推理
results = model("https://ultralytics.com/images/bus.jpg", save=True)

# 3. 打印结果
print("推理完成")
print(results[0])

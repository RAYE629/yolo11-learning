from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
IMAGE_SOURCE = "https://ultralytics.com/images/bus.jpg"

def main():
    model = YOLO(MODEL_PATH)

    results = model(
        IMAGE_SOURCE,
        save=True,
        conf=0.5,
        project="outputs",
        name="predict_conf_05",
    )

    result = results[0]

    print("推理完成")
    print("检测框数量:", len(result.boxes))
    print("类别名称:", result.names)

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = result.names[cls_id]
        print(f"{name}: {conf:.3f}")

if __name__ == "__main__":
    main()
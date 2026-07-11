from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
IMAGE_SOURCE = "https://ultralytics.com/images/bus.jpg"


def main():
    print("Step 1: 加载模型")
    model = YOLO(MODEL_PATH)

    print("Step 2: 开始推理")
    results = model(
        IMAGE_SOURCE,
        save=True,
        conf=0.5,
        project="outputs",
        name="chapter1_predict",
    )

    print("Step 3: 读取结果")
    result = results[0]

    print("原图尺寸:", result.orig_shape)
    print("检测框数量:", len(result.boxes))
    print("保存目录:", result.save_dir)

    for index, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        class_name = result.names[cls_id]

        print(
            f"{index}: "
            f"class={class_name}, "
            f"conf={confidence:.3f}, "
            f"xyxy={xyxy}"
        )


if __name__ == "__main__":
    main()
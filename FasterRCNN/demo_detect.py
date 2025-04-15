# demo_detect.py

from fasterrcnn_infer import load_model, detect_objects, detect_and_plot

# 设置路径与参数
image_path = "./images/test.jpg"  # 👈 替换为你的一张测试图片
weights_path = "best_model_distance.pth"
device = "cuda"  # 如果没有GPU，可设为 "cpu"


id_to_name = {
    0: 'suitcase',
    1: 'person',
    2: 'table',
    3: 'chair'
}


# 加载模型
model = load_model(weights_path=weights_path, num_classes=5, device=device)

# 推理图像
results = detect_and_plot(image_path, model, id_to_name=id_to_name, save_path="result.jpg")

# 打印结果
print("🎯 识别结果：")
for i, det in enumerate(results):
    x, y, cls_id, conf, w, h, dist = det
    print(f"目标{i+1}: 类别ID={cls_id}, 置信度={conf:.2f}, 距离={dist:.2f}m, 位置=({x:.1f},{y:.1f}), 尺寸=({w:.1f},{h:.1f})")

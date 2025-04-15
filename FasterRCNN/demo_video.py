import cv2
import time
from fasterrcnn_infer import load_model, detect_objects

# === 参数设置 ===
input_video = "video.avi"     # 你的原始视频
output_video = "output_video.avi"  # 推理后输出的视频
device = "cuda"  # or "cpu"
score_thresh = 0.5

id_to_name = {
    0: 'suitcase',
    1: 'person',
    2: 'table',
    3: 'chair'
}

# === 加载模型 ===
model = load_model("best_model_distance.pth", num_classes=5, device=device)

# === 打开视频
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频文件: {input_video}")

# 获取视频信息
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建输出视频写入器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"🎥 正在处理视频: {input_video}（共 {total_frames} 帧）")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === 推理开始计时
    start = time.time()

    # === 保存当前帧为临时图像
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, frame)

    # 推理
    results = detect_objects(temp_path, model, device=device, score_thresh=score_thresh)

    # === 绘制检测框
    for det in results:
        x, y, cls_id, conf, w, h, dist = det
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = id_to_name.get(cls_id, f"class {cls_id}")
        label_text = f"{label} | {dist:.2f}m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # === 记录推理时间
    end = time.time()
    interval = end - start
    print(f"🕒 帧 {frame_idx}: 推理时间 {interval:.3f} 秒")

    # 写入输出视频
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"\n✅ 已保存检测结果视频为 {output_video}")

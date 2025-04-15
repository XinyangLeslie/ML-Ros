#!/usr/bin/env python
# coding: utf-8

# ## 划分数据集

import os
import random
import shutil

# 设置路径
base_dir = os.getcwd()
image_dir = os.path.join(base_dir, 'images_new')
label_dir = os.path.join(base_dir, 'label_distance')

# 目标文件夹
train_img_dir = os.path.join(base_dir, 'train/images')
train_lbl_dir = os.path.join(base_dir, 'train/labels')
val_img_dir = os.path.join(base_dir, 'val/images')
val_lbl_dir = os.path.join(base_dir, 'val/labels')

# 创建文件夹
for folder in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(folder, exist_ok=True)

# 获取所有图片文件名（确保按 jpg 后缀）
all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
random.shuffle(all_images)

# 8:2 划分
split_idx = int(len(all_images) * 0.8)
train_files = all_images[:split_idx]
val_files = all_images[split_idx:]

def copy_files(file_list, target_img_dir, target_lbl_dir):
    for img_file in file_list:
        label_file = img_file.replace('.jpg', '.txt')
        src_img = os.path.join(image_dir, img_file)
        src_lbl = os.path.join(label_dir, label_file)
        dst_img = os.path.join(target_img_dir, img_file)
        dst_lbl = os.path.join(target_lbl_dir, label_file)

        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

# 拷贝
copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(val_files, val_img_dir, val_lbl_dir)

print(f"✅ 数据划分完成：训练集 {len(train_files)} 张，验证集 {len(val_files)} 张")


# ## 检查数据集

# In[1]:


import os
import cv2
import matplotlib.pyplot as plt

# 配置路径
image_dir = './train/images'
label_dir = './train/labels'

id_to_name = {
    0: 'suitcase',
    1: 'person',
    2: 'table',
    3: 'chair'
}

# 选择一张图
idx = 19
image_files = sorted(os.listdir(image_dir))
img_name = image_files[idx]
img_path = os.path.join(image_dir, img_name)
label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))

# 读取图片
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# 读取标签并绘图
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls_id, cx, cy, bw, bh, dist = parts
            cls_id = int(float(cls_id))

            label = id_to_name.get(cls_id, f"class {cls_id}")
            cx, cy, bw, bh = float(cx)*w, float(cy)*h, float(bw)*w, float(bh)*h
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            x2 = int(cx + bw/2)
            y2 = int(cy + bh/2)

            # 绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 类别 + 距离标签
            label_text = f"{label} | {float(dist):.2f}m"
            cv2.putText(image, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# 显示图像
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.title(f"Image: {img_name}")
plt.axis('off')
plt.show()


# In[3]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

# ✅ Dataset 类
class YoloToFRCNNDataset(Dataset):
    def __init__(self, image_dir, label_dir, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.train = train

        # 类别映射：YOLO原始ID => 连续索引
        self.id_to_name = {
            0: 'suitcase',
            1: 'person',
            2: 'table',
            3: 'chair'
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        boxes = []
        labels = []
        distances = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    cls_id, cx, cy, bw, bh, dist = parts
                    cls_id = int(float(cls_id))  # 转为整数

                    label = cls_id

                    # 反归一化
                    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2

                    boxes.append([x1, y1, x2, y2])
                    labels.append(label)
                    distances.append(float(dist))

        # 数据增强
        if self.train:
            if random.random() < 0.5:
                image = F.hflip(image)
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    boxes[i] = [w - x2, y1, w - x1, y2]
            image = F.adjust_brightness(image, 1 + (random.random() - 0.5) * 0.4)
            image = F.adjust_contrast(image, 1 + (random.random() - 0.5) * 0.4)

        image_tensor = T.ToTensor()(image)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "distances": torch.tensor(distances, dtype=torch.float32),
        }

        return image_tensor, target


# In[5]:


import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN

class FasterRCNNWithDistance(FasterRCNN):
    def __init__(self, backbone, num_classes):
        super().__init__(backbone, num_classes)
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.distance_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, images, targets=None):
        if self.training:
            # 🔧 原始损失
            loss_dict = super().forward(images, targets)

            # 🔧 特征提取
            features = self.backbone(torch.stack(images))  # OrderedDict[str, Tensor]

            # 🔧 所有 GT 框
            all_gt_boxes = [t["boxes"] for t in targets]
            image_shapes = [img.shape[1:] for img in images]

            # 🔧 RoI 特征提取
            box_features = self.roi_heads.box_roi_pool(features, all_gt_boxes, image_shapes)
            box_features = self.roi_heads.box_head(box_features)

            # 🔧 距离预测 + loss
            pred_distances = self.distance_head(box_features).squeeze(1)
            gt_distances = torch.cat([t["distances"] for t in targets]).to(pred_distances.device)
            distance_loss = nn.functional.smooth_l1_loss(pred_distances, gt_distances)

            loss_dict["loss_distance"] = distance_loss
            return loss_dict

        else:
            # 🧊 推理模式
            detections = super().forward(images)

            # 🔧 提取 features（OrderedDict[str, Tensor]）
            features = self.backbone(torch.stack(images))

            all_boxes = [d["boxes"] for d in detections]
            image_shapes = [img.shape[1:] for img in images]

            if sum(len(b) for b in all_boxes) == 0:
                # 没有任何预测框，直接返回空距离
                for det in detections:
                    det["distances"] = torch.tensor([]).to(images[0].device)
                return detections

            # 🔧 提取 RoI 特征
            box_features = self.roi_heads.box_roi_pool(features, all_boxes, image_shapes)
            box_features = self.roi_heads.box_head(box_features)
            pred_distances = self.distance_head(box_features).squeeze(1)

            # 🔧 将距离结果拆分到每张图像
            start = 0
            for i in range(len(detections)):
                num_boxes = len(detections[i]["boxes"])
                if num_boxes == 0:
                    detections[i]["distances"] = torch.tensor([]).to(images[0].device)
                else:
                    detections[i]["distances"] = pred_distances[start:start+num_boxes].detach().cpu()
                    start += num_boxes

            return detections


# In[7]:


import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2

def visualize_prediction(model, image_path, device, id_to_name, score_thresh=0.5):
    # 加载图片
    orig = cv2.imread(image_path)
    image = cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # 图像预处理
    transform = T.Compose([
        T.ToTensor()
    ])
    img_tensor = transform(image).to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs['boxes'].cpu()
    labels = outputs['labels'].cpu()
    scores = outputs['scores'].cpu()
    distances = outputs['distances'].cpu() if 'distances' in outputs else None

    # 绘制框
    for box, label, score, dist in zip(boxes, labels, scores, distances):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        name = id_to_name.get(label.item(), f"cls {label.item()}")
        dist_str = f"{dist:.2f}m" if distances is not None else ""
        label_str = f"{name} | {dist_str}"

        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig, label_str, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 显示图像
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {image_path}")
    plt.axis('off')
    plt.show()


# In[9]:


#from model_with_distance import FasterRCNNWithDistance
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# 类别映射（你的连续 ID）
id_to_name = {
    0: 'suitcase',
    1: 'person',
    2: 'table',
    3: 'chair'
}

# 模型加载
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = resnet_fpn_backbone('resnet101', pretrained=False)
model = FasterRCNNWithDistance(backbone, num_classes=5).to(device)
model.load_state_dict(torch.load("best_model_distance.pth", map_location=device))
model.eval()

# 推理一张图片
image_path = "./val/images/frame_00000_2.jpg"  # 👈 替换为你的一张图
visualize_prediction(model, image_path, device, id_to_name)


# In[11]:


def collate_fn(batch):
    return tuple(zip(*batch))

# ✅ 模型构建（ResNet101 + FPN）
def get_model(num_classes):
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    model = FasterRCNNWithDistance(backbone, num_classes)
    return model


# ✅ 验证函数（每轮评估）
import torch
from torchvision.ops import box_iou

@torch.no_grad()
def evaluate_model(model, val_loader, device="cuda", iou_threshold=0.6, score_threshold=0.5):
    model.eval()
    model.to(device)

    total_preds = 0
    total_gts = 0
    correct = 0
    correct_total_images = 0
    distance_errors = []  # ✅ 用于记录预测与 GT 距离的差异

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)

        for pred, target in zip(outputs, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            pred_dists = pred['distances'] if 'distances' in pred else None

            gt_boxes = target['boxes']
            gt_labels = target['labels']
            gt_dists = target['distances']

            keep = pred_scores > score_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            if pred_dists is not None:
                keep = keep.to(pred_dists.device)  # ✅ 保证 keep 和 pred_dists 在同一设备
                pred_dists = pred_dists[keep]

            total_preds += len(pred_boxes)
            total_gts += len(gt_boxes)

            matched = 0
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes)
                for i in range(len(pred_boxes)):
                    max_iou, gt_idx = ious[i].max(0)
                    if max_iou > iou_threshold and pred_labels[i] == gt_labels[gt_idx]:
                        correct += 1
                        matched += 1
                        ious[:, gt_idx] = 0  # 防止重复匹配

                        # ✅ 计算距离误差
                        if pred_dists is not None and gt_idx < len(gt_dists):
                            pred_dist = pred_dists[i].item()
                            gt_dist = gt_dists[gt_idx].item()
                            distance_errors.append(abs(pred_dist - gt_dist))

            if matched > 0:
                correct_total_images += 1

    precision = correct / total_preds if total_preds > 0 else 0
    recall = correct / total_gts if total_gts > 0 else 0
    accuracy = correct_total_images / len(val_loader.dataset)

    avg_distance_error = sum(distance_errors) / len(distance_errors) if distance_errors else -1

    return precision, recall, accuracy, avg_distance_error


# ✅ 数据 & 模型准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = YoloToFRCNNDataset('./train/images', './train/labels', train=True)
val_dataset = YoloToFRCNNDataset('./val/images', './val/labels', train=False)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
model = get_model(num_classes=5)
model.to(device)
print(torch.cuda.is_available())  # True 才说明有 GPU
print(device)                     # 应该是 'cuda'


# ✅ 加载已保存的权重和 best_loss
best_loss = float('inf')
# if os.path.exists("best_model_resnet101.pth"):
#     model.load_state_dict(torch.load("best_model_resnet101.pth", map_location=device))
#     print("✅ Loaded best_model_resnet101.pth")

if os.path.exists("best_loss.txt"):
    with open("best_loss.txt", "r") as f:
        best_loss = float(f.read())
    print(f"📉 Loaded previous best loss: {best_loss:.4f}")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# ✅ 训练 + 评估主循环
num_epochs = 50
loss_list = []
precision_list = []
recall_list = []
acc_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"📦 Epoch {epoch+1}/{num_epochs}")

    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
        # ✅ 过滤掉 boxes 数量为 0 的样本
        filtered_images = []
        filtered_targets = []
        for img, tgt in zip(images, targets):
            if tgt['boxes'].numel() > 0:
                filtered_images.append(img)
                filtered_targets.append(tgt)
    
        if len(filtered_images) == 0:
            continue  # 全是空标签，跳过该 batch
    
        loss_dict = model(filtered_images, filtered_targets)
        #print(loss_dict)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        pbar.set_postfix({k: f"{v.item():.4f}" for k, v in loss_dict.items()})

    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)
    lr_scheduler.step(avg_loss)

    # ✅ 每轮评估
    precision, recall, acc, dist_err = evaluate_model(model, val_loader, device=device)

    print(f"\n📊 Epoch {epoch+1}: Avg Loss={avg_loss:.4f} | 🎯 Precision={precision:.4f} | 🔁 Recall={recall:.4f} | ✅ Accuracy={acc:.4f} | 📏 AvgDistError={dist_err:.2f}m\n")

    precision_list.append(precision)
    recall_list.append(recall)
    acc_list.append(acc)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model_distance.pth")
        with open("best_loss.txt", "w") as f:
            f.write(str(best_loss))
        print("✅ Saved best model.")

# ✅ 训练完成，绘制多图对比
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(loss_list, marker='o')
plt.title("Loss")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(precision_list, marker='o', label='Precision')
plt.plot(recall_list, marker='s', label='Recall')
plt.title("Precision / Recall")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(acc_list, marker='^', color='green')
plt.title("Accuracy (image level)")
plt.grid(True)

plt.tight_layout()
plt.show()


# In[13]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[15]:


print(device)


# In[ ]:





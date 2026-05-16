import os
import random

import torch
import cv2
import numpy as np
import torchvision.ops as ops
from utils import cells_to_bboxes 
import config 
from model import YOLOv3

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# def predict_image(image_path, model, device="cuda", threshold=0.6, iou_threshold=0.3, agnostic_nms=False):
#     model.eval()

#     # 1. ĐỌC VÀ TIỀN XỬ LÝ ẢNH
#     original_image = cv2.imread(image_path)
#     if original_image is None:
#         raise ValueError(f"Không tìm thấy ảnh tại {image_path}")
    
#     img_h, img_w, _ = original_image.shape
#     image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
#     # Resize về 416x416 (Nếu bạn dùng Albumentations thì có thể gọi transform ở đây)
#     image_resized = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
#     image_resized = image_resized / 255.0
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])

#     x = (image_resized - mean) / std

#     # Chuyển thành Tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
#     x = torch.from_numpy(x).permute(2, 0, 1).float()
    

#     x = x.unsqueeze(0).to(device)

#     # 2. ĐƯA QUA MODEL
#     with torch.no_grad():
#         predictions = model(x)

#     # 3. GIẢI MÃ BBOXES
#     bboxes = []
#     for i in range(3):
#         S = predictions[i].shape[2]
#         anchor = torch.tensor([*config.ANCHORS[i]]).to(device) * S
#         # Lấy định dạng corners (x1, y1, x2, y2) để dễ dùng NMS
#         boxes_scale_i = cells_to_bboxes(
#             predictions[i], anchor, S=S, is_preds=True, output_format='xyxy'
#         )
#         bboxes += boxes_scale_i[0] # Lấy index 0 vì batch_size = 1

#     # 4. LỌC NMS VÀ NGƯỠNG
#     boxes = torch.tensor(bboxes).to(device)
#     class_preds = boxes[:, 0]
#     scores = boxes[:, 1]
#     coords = boxes[:, 2:] # (x1, y1, x2, y2) đã được chuẩn hóa (0 -> 1)

#     w = coords[:, 2] - coords[:, 0]
#     h = coords[:, 3] - coords[:, 1]

#     # Lọc box có điểm số > threshold và kích thước hợp lệ
#     keep_mask = (scores > threshold) & (w > 0.01) & (h > 0.01)
#     scores = scores[keep_mask]
#     coords = coords[keep_mask]
#     class_preds = class_preds[keep_mask]


#     TOPK_PRE_NMS = config.PRE_NMS_TOP_K

#     topk_idx = scores.argsort(descending=True)[:TOPK_PRE_NMS]

#     scores = scores[topk_idx]
#     coords = coords[topk_idx]
#     class_preds = class_preds[topk_idx]
#     if len(coords) > 0:
#         if agnostic_nms:
#             # CLASS-AGNOSTIC NMS: Chỉ quan tâm đến tọa độ và điểm số
#             keep_indices = ops.nms(coords, scores, iou_threshold)
#         else:
#             # BATCHED NMS: Quan tâm thêm class_preds
#             # Chỉ xóa nếu 2 box đè lên nhau VÀ có cùng class
#             keep_indices = ops.batched_nms(coords, scores, class_preds, iou_threshold)
        
#         TOK_KEEP = 100 # Số lượng box tối đa sau NMS (có thể điều chỉnh)
#         post_nms_keep_indices = keep_indices[
#             scores[keep_indices].argsort(descending=True)
#         ]
#         keep_indices = post_nms_keep_indices[:TOK_KEEP]
#         coords = coords[keep_indices]
#         scores = scores[keep_indices]
#         class_preds = class_preds[keep_indices]
    

#     print(f"Phát hiện {len(coords)} đối tượng sau NMS")
#     # 5. VẼ LÊN ẢNH GỐC
#     for i in range(len(coords)):
#         box = coords[i].cpu().numpy()
#         class_id = int(class_preds[i].item())
#         score = scores[i].item()

#         # Dịch ngược tọa độ (0-1) về kích thước ảnh gốc
#         x1 = int(box[0] * img_w)
#         y1 = int(box[1] * img_h)
#         x2 = int(box[2] * img_w)
#         y2 = int(box[3] * img_h)

#         # Vẽ hình chữ nhật
#         color = (0, 255, 0) # Màu xanh lá
#         cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
        
#         # In tên class và độ tự tin
#         label_text = f"{VOC_CLASSES[class_id]}: {score:.2f}"
#         cv2.putText(original_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # Lưu hoặc hiển thị ảnh
#     cv2.imshow("YOLOv3 Prediction", original_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



import xml.etree.ElementTree as ET

def get_ground_truth_from_xml(xml_path):
    """Đọc tọa độ Ground Truth từ file XML của Pascal VOC"""
    if not os.path.exists(xml_path):
        return []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_boxes = []
    
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        # Chuyển về tọa độ pixel tuyệt đối
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        gt_boxes.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})
    return gt_boxes

def draw_professional_box(image, x1, y1, x2, y2, label_text, color, font_scale=0.5, thickness=2):
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = max(1, thickness - 1)
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
    txt_box_y1 = y1 - text_h - baseline - 5
    if txt_box_y1 < 0: txt_box_y1 = y1 + 1
    cv2.rectangle(image, (x1, txt_box_y1), (x1 + text_w + 10, txt_box_y1 + text_h + baseline + 5), color, -1)
    cv2.putText(image, label_text, (x1 + 5, txt_box_y1 + text_h + 3), font, 
                font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

def predict_image(image_path, model, device="cuda", threshold=0.5, iou_threshold=0.4):
    model.eval()
    
    # 1. Đọc ảnh và tìm file XML tương ứng
    original_image = cv2.imread(image_path)
    gt_image = original_image.copy()
    img_h, img_w, _ = original_image.shape
    
    xml_path = image_path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
    gt_boxes = get_ground_truth_from_xml(xml_path)

    # 2. Tiền xử lý cho Model
    image_input = cv2.resize(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image_input = (image_input / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.from_numpy(image_input).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # 3. Dự đoán
    with torch.no_grad():
        out = model(x)
    bboxes = []
    for i in range(3):
        S = out[i].shape[2]
        anchor = torch.tensor([*config.ANCHORS[i]]).to(device) * S
        bboxes += cells_to_bboxes(out[i], anchor, S=S, is_preds=True, output_format='xyxy')[0]

    # 4. NMS
    boxes = torch.tensor(bboxes).to(device)
    scores, class_preds, coords = boxes[:, 1], boxes[:, 0], boxes[:, 2:]
    keep_mask = (scores > threshold)
    scores, coords, class_preds = scores[keep_mask], coords[keep_mask], class_preds[keep_mask]
    keep = ops.batched_nms(coords, scores, class_preds, iou_threshold)
    coords, scores, class_preds = coords[keep], scores[keep], class_preds[keep]

    # 5. Vẽ Ground Truth (Bên trái)
    np.random.seed(42)
    colors = [tuple(int(c) for c in color) for color in np.random.randint(0, 255, size=(len(VOC_CLASSES), 3))]
    
    for gt in gt_boxes:
        cid = VOC_CLASSES.index(gt['label'])
        bx = gt['bbox']
        draw_professional_box(gt_image, bx[0], bx[1], bx[2], bx[3], f"{gt['label']}", colors[cid], thickness=2)

    # 6. Vẽ Dự đoán (Bên phải)
    for i in range(len(coords)):
        box, cid, sc = coords[i].cpu().numpy(), int(class_preds[i].item()), scores[i].item()
        x1, y1 = int(box[0] * img_w), int(box[1] * img_h)
        x2, y2 = int(box[2] * img_w), int(box[3] * img_h)
        draw_professional_box(original_image, x1, y1, x2, y2, f"{VOC_CLASSES[cid]} {sc:.2f}", colors[cid], thickness=2)

    # 7. Ghép ảnh Side-by-Side
    combined_img = np.hstack((gt_image, original_image))
    
    # Thêm tiêu đề
    # cv2.putText(combined_img, "GROUND TRUTH", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    # cv2.putText(combined_img, "DETECTIONS", (img_w + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("VOC 2007 Comparison: GT vs Prediction", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    pth = "VOCdevkit/VOC2007_test/JPEGImages"

    print(f"Using device: {config.DEVICE}")
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # Priority: BEST_WEIGHTS_FILE > CHECKPOINT_FILE
    weight_path = config.CHECKPOINT_FILE
    
    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}...")
        checkpoint = torch.load(weight_path, map_location=config.DEVICE)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No weights found. Evaluating random model.")

    
    for img_name in random.sample(os.listdir(pth), k=20):  # Chọn ngẫu nhiên 20 ảnh để dự đoán
        img_path = os.path.join(pth, img_name)
        predict_image(img_path, model)

if __name__ == "__main__":
    main()
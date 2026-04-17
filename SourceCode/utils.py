import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import config

from collections import Counter
from tqdm import tqdm

def iou_width_height(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes1, boxes2, box_format="xywh"):
    # boxes shape (..., 4)

    if box_format == "xywh":
        # (x_center, y_center, width, height) → (x1, y1, x2, y2)
        box1_x1 = boxes1[..., 0] - boxes1[..., 2] / 2
        box1_y1 = boxes1[..., 1] - boxes1[..., 3] / 2
        box1_x2 = boxes1[..., 0] + boxes1[..., 2] / 2
        box1_y2 = boxes1[..., 1] + boxes1[..., 3] / 2

        box2_x1 = boxes2[..., 0] - boxes2[..., 2] / 2
        box2_y1 = boxes2[..., 1] - boxes2[..., 3] / 2
        box2_x2 = boxes2[..., 0] + boxes2[..., 2] / 2
        box2_y2 = boxes2[..., 1] + boxes2[..., 3] / 2

    elif box_format == "xyxy":
        # (x1, y1, x2, y2)
        box1_x1 = boxes1[..., 0]
        box1_y1 = boxes1[..., 1]
        box1_x2 = boxes1[..., 2]
        box1_y2 = boxes1[..., 3]

        box2_x1 = boxes2[..., 0]
        box2_y1 = boxes2[..., 1]
        box2_x2 = boxes2[..., 2]
        box2_y2 = boxes2[..., 3]

    else:
        raise ValueError("box_format must be 'xywh' or 'xyxy'")
        
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = (box1_x2 - box1_x1).clamp(0) * (box1_y2 - box1_y1).clamp(0)
    box2_area = (box2_x2 - box2_x1).clamp(0) * (box2_y2 - box2_y1).clamp(0)

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def check_class_accuracy(model, loader, threshold):
    model.eval()

    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    total_iou = 0
    total_loc = 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)

        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)

            obj = y[i][..., 0] == 1
            noobj = y[i][..., 0] == 0

            # ------------------
            # CLASS ACCURACY
            # ------------------
            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1)
                == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            # ------------------
            # OBJECTNESS ACCURACY
            # ------------------
            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold

            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)

            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

            # ------------------
            # LOCALIZATION (IoU)
            # ------------------

            # predicted boxes (midpoint)
            pred_boxes = out[i][..., 1:5]
            pred_boxes[..., 0:2] = torch.sigmoid(pred_boxes[..., 0:2])
            anchors = (torch.tensor(config.ANCHORS[i])).to(config.DEVICE)

            S = out[i].shape[2] # 13
            anchors = anchors * S
            anchors = anchors.reshape(1, 3, 1, 1, 2)
            pred_boxes[..., 2:] = torch.exp(pred_boxes[..., 2:]) * anchors # pred box in feature space

            # chuyển sang normalized

            cell_indices = (
                torch.arange(S)
                .repeat(out[i].shape[0], 3, S, 1)
                .unsqueeze(-1)
                .to(config.DEVICE)
            )

            x_center = (pred_boxes[..., 0:1] + cell_indices) / S
            y_center = (pred_boxes[..., 1:2] + cell_indices.permute(0,1,3,2,4)) / S
            w_h = pred_boxes[..., 2:4] / S

            pred_boxes = torch.cat([x_center, y_center, w_h], dim=-1)
            # ground truth boxes
            true_boxes = y[i][..., 1:5]
            cell_indices = (
                torch.arange(S)
                .repeat(out[i].shape[0], 3, S, 1)
                .unsqueeze(-1)
                .to(config.DEVICE)
            )

            x_true = (true_boxes[..., 0:1] + cell_indices) / S
            y_true = (true_boxes[..., 1:2] + cell_indices.permute(0,1,3,2,4)) / S
            w_true = true_boxes[..., 2:3] / S
            h_true = true_boxes[..., 3:4] / S

            true_boxes = torch.cat([x_true, y_true, w_true, h_true], dim=-1)
            if obj.sum() > 0:
                ious = intersection_over_union(
                    pred_boxes[obj],
                    true_boxes[obj],
                    box_format="xywh",
                )

                total_iou += ious.sum()
                total_loc += ious.numel()
    print(pred_boxes[obj][:2])
    print(true_boxes[obj][:2])
    print(f"Class accuracy: {(correct_class/(tot_class_preds+1e-16))*100:.2f}%")
    print(f"No obj accuracy: {(correct_noobj/(tot_noobj+1e-16))*100:.2f}%")
    print(f"Obj accuracy: {(correct_obj/(tot_obj+1e-16))*100:.2f}%")

    if total_loc > 0:
        print(f"Mean IoU (localization): {(total_iou/total_loc):.4f}")

    model.train()


def cells_to_bboxes(predictions, anchors, S, is_preds=True, output_format = 'xywh'):
    """
    predictions: Tensor dự đoán từ model HOẶC tensor Target (Ground Truth)
    is_preds: True nếu là dự đoán, False nếu là Target
    """
    B = predictions.shape[0]
    num_anchors = len(anchors)

    if is_preds:
        box_predictions = predictions[..., 1:5] # Lấy [x, y, w, h]
        objectness = torch.sigmoid(predictions[..., 0:1]) # Lấy [obj]
        class_probs = torch.sigmoid(predictions[..., 5:]) # Xác suất 20 classes
        
        best_class_prob, best_class = torch.max(class_probs, dim=-1, keepdim=True)
        final_scores = objectness * best_class_prob

        # Biến đổi thô (logits) thành tọa độ
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:4] = torch.exp(box_predictions[..., 2:4]) * anchors
        
    else:
        # Tensor Target có cấu trúc: [obj, x_offset, y_offset, w_grid, h_grid, class_index]
        box_predictions = predictions[..., 1:5] 
        final_scores = predictions[..., 0:1] # Điểm tự tin của GT luôn là 1 (hoặc 0)
        best_class = predictions[..., 5:6]   # Chứa sẵn ID của class (ví dụ: 14 là Person)
        
        # Ground Truth KHÔNG CẦN dùng sigmoid hay exp vì tọa độ đã được chuẩn hóa sẵn!

    # Tạo mảng grid chứa tọa độ các ô lưới
    grid = torch.arange(S).repeat(B, num_anchors, S, 1).unsqueeze(-1).to(predictions.device)

    # Đưa x, y, w, h về (0.0 đến 1.0)
    x = (box_predictions[..., 0:1] + grid) / S
    y = (box_predictions[..., 1:2] + grid.permute(0, 1, 3, 2, 4)) / S
    w_h = box_predictions[..., 2:4] / S;
    if output_format == 'xywh':
        converted_bboxes = torch.cat((best_class, final_scores, x, y, w_h), dim=-1).reshape(B, num_anchors * S * S, 6)
    else:
        x1 = x - w_h[..., 0:1] / 2
        y1 = y - w_h[..., 1:2] / 2
        x2 = x + w_h[..., 0:1] / 2
        y2 = y + w_h[..., 1:2] / 2
        # clamp
        x1 = x1.clamp_(0, 1)
        y1 = y1.clamp_(0, 1)
        x2 = x2.clamp_(0, 1)
        y2 = y2.clamp_(0, 1)
        converted_bboxes = torch.cat((best_class, final_scores, x1, y1, x2, y2), dim=-1).reshape(B, num_anchors * S * S, 6)
    return converted_bboxes.tolist()
    # Kết quả trả về: [Class, Score, X_center, Y_center, Width, Height]

import torchvision.ops as ops
def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    device="cuda",
):
    print("Geting bboxes from testloader...")
    # make sure model is in eval before get bboxes
    model.eval()

    all_pred_boxes = []
    all_true_boxes = []
    train_idx = 0
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        true_bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True, output_format='xyxy'
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box


            true_boxes = cells_to_bboxes(
                labels[i], anchor, S=S, is_preds=False, output_format='xyxy' # here
            )
            for idx, box, in enumerate(true_boxes):
                true_bboxes[idx] += box


        for idx in range(batch_size):
            if len(bboxes[idx]) == 0:
                continue

            boxes = torch.tensor(bboxes[idx]).to(device) # (N, 6) class, score, x,y,w,h

            class_preds = boxes[:, 0]
            scores = boxes[:, 1]
            coords = boxes[:, 2:]   # x,y,w,h hoặc x1,y1,x2,y2
            

            if coords.shape[0] == 0:
                continue
            

            pred_coords = []
            pred_scores = []
            pred_labels = []
            # keep_mask = torch.ones(coords.shape[0], dtype=torch.bool).to(device)

            # pre-nms filtering: 400 box per class
            for label in range(config.NUM_CLASSES):
                label_mask = class_preds == label
                scores_label = scores[label_mask] # (N_label,)
                coords_label = coords[label_mask] # (N_label, 4)
                class_preds_label = class_preds[label_mask] # (N_label,)
                assert scores_label.shape[0] == coords_label.shape[0] == class_preds_label.shape[0]

                keep_idxs = scores_label > threshold 

                scores_label = scores_label[keep_idxs]
                coords_label = coords_label[keep_idxs]
                class_preds_label = class_preds_label[keep_idxs]
                if coords_label.shape[0] == 0:
                    continue
                
                score, top_k_idxs = scores_label.topk(min(config.PRE_NMS_TOP_K, scores_label.shape[0]))
                coords_label = coords_label[top_k_idxs]
                class_preds_label = class_preds_label[top_k_idxs]
                pred_coords.append(coords_label)
                pred_scores.append(score)
                pred_labels.append(class_preds_label)
            
            if len(pred_coords) == 0: continue
            pred_coords = torch.cat(pred_coords, dim=0)
            pred_scores = torch.cat(pred_scores, dim=0)
            pred_labels = torch.cat(pred_labels, dim=0)

            keep_indices = ops.batched_nms(pred_coords, pred_scores, pred_labels, iou_threshold)
            post_nms_keep_indices = keep_indices[
                pred_scores[keep_indices].sort(descending=True)[1]
            ]
            TOK_K = config.BOXES_PER_IMAGE
            keep = post_nms_keep_indices[:TOK_K] # keep top 100 after nms

            pred_coords = pred_coords[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            selected = torch.cat(S
                [
                    pred_labels.unsqueeze(1),
                    pred_scores.unsqueeze(1),
                    pred_coords,
                ],
                dim=1,
            )

            nms_boxes = selected.cpu().tolist()
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
             

    print('Done getting bboxes.')
    model.train()
    return all_pred_boxes, all_true_boxes



def compute_metrics(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    score_threshold=0.5, 
    box_format="xyxy",
    num_classes=20,
):
    average_precisions = []
    per_class_metrics = {}
    
    total_TP_all = 0
    total_FP_all = 0
    total_FN_all = 0
    
    epsilon = 1e-6

    # Chuyển đổi sang tensor nếu input đang là list để tăng tốc
    if isinstance(pred_boxes, list):
        pred_boxes = torch.tensor(pred_boxes)
    if isinstance(true_boxes, list):
        true_boxes = torch.tensor(true_boxes)

    for c in range(num_classes):
        detections = pred_boxes[pred_boxes[:, 1] == c]
        ground_truths = true_boxes[true_boxes[:, 1] == c]

        total_true_bboxes = ground_truths.shape[0]
        if total_true_bboxes == 0:
            continue

        # amount_bboxes để đánh dấu GT đã được khớp chưa
        amount_bboxes = Counter(ground_truths[:, 0].tolist())
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sắp xếp dự đoán theo confidence (cột index 2) giảm dần
        detections = detections[detections[:, 2].argsort(descending=True)]

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = ground_truths[ground_truths[:, 0] == detection[0]]

            best_iou = 0
            best_gt_idx = -1
            img_idx = detection[0].item()

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    detection[3:], gt[3:], box_format=box_format
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[img_idx][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[img_idx][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls_curve = TP_cumsum / (total_true_bboxes + epsilon)
        precisions_curve = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        
        # Thêm điểm đầu cho đồ thị
        p_c = torch.cat((torch.tensor([1.0]), precisions_curve))
        r_c = torch.cat((torch.tensor([0.0]), recalls_curve))

        AP = torch.trapz(p_c, r_c)
        average_precisions.append(AP)

        # Tính Precision/Recall tại score_threshold
        # Tìm những index mà tại đó confidence > score_threshold
        keep_indices = detections[:, 2] > score_threshold
        
        # Tổng số TP và FP tại ngưỡng score_threshold này
        # Chúng ta dùng mask keep_indices để lọc mảng TP, FP ban đầu
        tp_at_thresh = TP[keep_indices].sum().item()
        fp_at_thresh = FP[keep_indices].sum().item()
        
        # Tính P, R tại ngưỡng này
        p_class = tp_at_thresh / (tp_at_thresh + fp_at_thresh + epsilon)
        r_class = tp_at_thresh / (total_true_bboxes + epsilon)
        fn_class = total_true_bboxes - tp_at_thresh
        f1_class = 2 * p_class * r_class / (p_class + r_class + epsilon)

        per_class_metrics[c] = {
            "precision": p_class,
            "recall": r_class,
            "f1": f1_class,
        }

        # Cộng dồn để tính Micro-average (cũng tại ngưỡng score_threshold)
        total_TP_all += tp_at_thresh
        total_FP_all += fp_at_thresh
        total_FN_all += fn_class

    # Overall Metrics
    overall_precision = total_TP_all / (total_TP_all + total_FP_all + epsilon)
    overall_recall = total_TP_all / (total_TP_all + total_FN_all + epsilon)
    overall_f1 = 2 * overall_precision * overall_recall / (
        overall_precision + overall_recall + epsilon
    )

    return {
        "map": (sum(average_precisions) / len(average_precisions)).item() if average_precisions else 0.0,
        "per_class_metric": per_class_metrics,
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        },
    }
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

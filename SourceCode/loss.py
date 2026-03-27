import torch
import torch.nn as nn
from utils import *
import config
from dataset import get_train_test_loader

def build_targets(targets, anchors, S, num_anchors_per_scale=3, ignore_iou_thresh=0.5):

    """
        targets = (
            {
                'bboxes': [box1, box2,...],
                'class_labels': [label1, label2,...]
                'difficult': [0, 1, ....]
            }, -> image 1
            {
                image2 info
            }
        )
        anchors(normalize) = [
            [a1, a2, a3], [a4, a5, a6], [a7, a8, a9]
        ]
    """
    B = len(targets)
    device = config.DEVICE

    yolo_targets = [
        torch.zeros((B, num_anchors_per_scale, S[i], S[i], 6), device=device)
        for i in range(3)
    ]

    anchors = torch.tensor(anchors, device = device).reshape(-1, 2) # (9, 2)

    for b in range(B):

        boxes = targets[b]['bboxes']
        labels = targets[b]['labels']
        for box, class_label in zip(boxes, labels):

            x, y, w, h = box

            iou_anchors = iou_width_height(
                torch.tensor([w, h], device=device),
                anchors
            )

            anchor_indices = iou_anchors.argsort(descending=True)

            has_anchor = False

            for anchor_idx in anchor_indices:

                scale_idx = anchor_idx // num_anchors_per_scale
                anchor_on_scale = anchor_idx % num_anchors_per_scale

                S_scale = S[scale_idx]
                # cell chịu trách nhiệm cho box ở grid space
                i = int(S_scale * y)
                j = int(S_scale * x)

                anchor_taken = yolo_targets[scale_idx][b, anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor:

                    yolo_targets[scale_idx][b, anchor_on_scale, i, j, 0] = 1

                    x_cell = S_scale * x - j
                    y_cell = S_scale * y - i

                    w_cell = w * S_scale
                    h_cell = h * S_scale

                    yolo_targets[scale_idx][b, anchor_on_scale, i, j, 1:5] = \
                        torch.tensor([x_cell, y_cell, w_cell, h_cell], device=device)

                    yolo_targets[scale_idx][b, anchor_on_scale, i, j, 5] = class_label

                    has_anchor = True

                elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:

                    yolo_targets[scale_idx][b, anchor_on_scale, i, j, 0] = -1

    return yolo_targets



def count_anchor_states(train_loader):
    total_pos = [0, 0, 0]
    total_neg = [0, 0, 0]
    total_ignore = [0, 0, 0]

    for (_, yolo_targets) in train_loader:
        for scale_idx, target in enumerate(yolo_targets):
            obj_mask = target[..., 0]

            total_pos[scale_idx] += (obj_mask == 1).sum().item()
            total_neg[scale_idx] += (obj_mask == 0).sum().item()
            total_ignore[scale_idx] += (obj_mask == -1).sum().item()

    # in kết quả cuối
    for scale_idx in range(3):
        print(f"Scale {scale_idx}:")
        print(f"  +1 (object):  {total_pos[scale_idx]}")
        print(f"   0 (no obj):  {total_neg[scale_idx]}")
        print(f"  -1 (ignore): {total_ignore[scale_idx]}")
        print("-" * 30)

    # tổng tất cả scale
    print("TOTAL:")
    print(f"  +1 (object):  {sum(total_pos)}")
    print(f"   0 (no obj):  {sum(total_neg)}")
    print(f"  -1 (ignore): {sum(total_ignore)}")
    

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, train_set, test_set = get_train_test_loader()

    _, target = train_set[0]
    # print(target)

    
    loss_fn = YOLOLoss()

    count_anchor_states(train_loader)
    # fake_predictions = [
    #     torch.randn(32, 3, config.S[0], config.S[0], 5 + 20, device=device),
    #     torch.randn(32, 3, config.S[1], config.S[1], 5 + 20, device=device),
    #     torch.randn(32, 3, config.S[2], config.S[2], 5 + 20, device=device)
    # ]
    # loss, _,_,_,_ = loss_fn(fake_predictions, targets)

    # print(loss)
if __name__ == "__main__":
    main()
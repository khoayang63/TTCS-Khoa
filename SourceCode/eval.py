import torch
import config
from model import YOLOv3
from dataset import get_train_test_loader
from utils import get_evaluation_bboxes, compute_metrics
from tqdm import tqdm
from loss import YOLOLoss
import os

@torch.no_grad()
def eval_fn(val_loader, model, loss_fn):
    model.eval()
    print("Evaluating Loss...")
    loop = tqdm(val_loader, leave=True)
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (x, targets) in enumerate(loop):
        x = x.to(config.DEVICE)
        targets = [t.to(config.DEVICE) for t in targets]

        with torch.cuda.amp.autocast():
            out = model(x)
            # unpack 5 values from loss_fn
            loss, _, _, _, _ = loss_fn(out, targets)

        running_loss += loss.item()
        num_batches += 1
        loop.set_postfix({"loss": f"{running_loss/num_batches:.3f}"})

    return running_loss / num_batches

def print_simple_table(metrics):
    header = f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}"
    separator = "-" * len(header)
    
    print(f"\n[ SUMMARY METRICS ]")
    print(f"mAP: {metrics['map']:.4f}")
    print(separator)
    print(header)
    print(separator)
    
    for class_id, m in metrics['per_class_metric'].items():
        print(f"{config.PASCAL_CLASSES[class_id]:<15} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f}")
        
    print(separator)
    o = metrics['overall']
    print(f"{'OVERALL':<15} | {o['precision']:<10.4f} | {o['recall']:<10.4f} | {o['f1']:<10.4f}")
    print(separator)

def main():
    print(f"Using device: {config.DEVICE}")
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # Priority: BEST_WEIGHTS_FILE > CHECKPOINT_FILE
    weight_path = config.BEST_WEIGHTS_FILE if os.path.exists(config.BEST_WEIGHTS_FILE) else config.CHECKPOINT_FILE
    
    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}...")
        checkpoint = torch.load(weight_path, map_location=config.DEVICE)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No weights found. Evaluating random model.")

    _, test_loader, _, _ = get_train_test_loader()
    
    # 1. Compute mAP and P/R/F1
    all_pred_boxes, all_true_boxes = get_evaluation_bboxes(
        test_loader, model, 
        iou_threshold=config.IOU_THRESH, 
        anchors=config.ANCHORS, 
        threshold=config.MAP_CONF_THRESH
    )
    
    print(f"Number of pred boxes: {len(all_pred_boxes)}")
    print(f"Number of gt boxes: {len(all_true_boxes)}")
    
    metrics = compute_metrics(all_pred_boxes, all_true_boxes)
    print_simple_table(metrics)

if __name__ == "__main__":
    main()

from dataset import YOLODataset, get_train_test_loader
from model import YOLOv3
from loss import YOLOLoss
import config
import torch
import os
from tqdm import tqdm
import warnings
from utils import *
from infer import predict_image

warnings.filterwarnings('ignore')

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, filename):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}.")

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}.")


def load_checkpoint(filename, model, optimizer, scheduler, scaler):
    print("Loading checkpoint...")
    checkpoint = torch.load(filename, map_location=config.DEVICE)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1

    print("Checkpoint loaded successfully.")
    return start_epoch

def train_fn(
    train_loader,
    model,
    optimizer,
    loss_fn,
    scaler,
    epoch,
    scheduler=None,
):
    model.train()
    print('Training epoch:', epoch)
    loop = tqdm(train_loader, leave=True)

    running_loss = 0.0
    running_box = 0.0
    running_obj = 0.0
    running_noobj = 0.0
    running_class = 0.0

    num_batches = 0

    warmup_epochs = 3
    warmup_iters = warmup_epochs * len(train_loader)
    base_lr = config.LEARNING_RATE

    for batch_idx, (x, targets) in enumerate(loop):
        x = x.to(config.DEVICE)
        targets = [t.to(config.DEVICE) for t in targets]

        global_step = epoch * len(train_loader) + batch_idx
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(x)
            loss, loss_box, loss_obj, loss_noobj, loss_class = loss_fn(out, targets)

            if torch.isnan(loss):
                print('Loss is becoming nan. Exiting')
                exit(0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_box += loss_box.item()
        running_obj += loss_obj.item()
        running_noobj += loss_noobj.item()
        running_class += loss_class.item()
        num_batches += 1

        # warmup / scheduler
        if global_step < warmup_iters:
            warmup_factor = global_step / warmup_iters
            lr = 1e-6 + base_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            if scheduler is not None:
                scheduler.step()

        # mean tạm thời (hiển thị)
        mean_loss = running_loss / num_batches
        mean_box = running_box / num_batches
        mean_obj = running_obj / num_batches
        mean_noobj = running_noobj / num_batches
        mean_class = running_class / num_batches

        loop.set_postfix({
            "loss": f"{mean_loss:.3f}",
            "box": f"{mean_box:.3f}",
            "obj": f"{mean_obj:.3f}",
            "noobj": f"{mean_noobj:.3f}",
            "cls": f"{mean_class:.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
        })

    mean_loss = running_loss / num_batches
    mean_box = running_box / num_batches
    mean_obj = running_obj / num_batches
    mean_noobj = running_noobj / num_batches
    mean_class = running_class / num_batches

    return mean_loss, mean_box, mean_obj, mean_noobj, mean_class

@torch.no_grad()
def eval_fn(
    val_loader,
    model,
    loss_fn,
):
    model.eval()
    print("Evaluating...")

    loop = tqdm(val_loader, leave=True)

    running_loss = 0.0
    running_box = 0.0
    running_obj = 0.0
    running_noobj = 0.0
    running_class = 0.0

    num_batches = 0

    for batch_idx, (x, targets) in enumerate(loop):
        x = x.to(config.DEVICE)
        targets = [t.to(config.DEVICE) for t in targets]

        with torch.cuda.amp.autocast():
            out = model(x)
            loss, loss_box, loss_obj, loss_noobj, loss_class = loss_fn(out, targets)

        running_loss += loss.item()
        running_box += loss_box.item()
        running_obj += loss_obj.item()
        running_noobj += loss_noobj.item()
        running_class += loss_class.item()
        num_batches += 1

        mean_loss = running_loss / num_batches
        mean_box = running_box / num_batches
        mean_obj = running_obj / num_batches
        mean_noobj = running_noobj / num_batches
        mean_class = running_class / num_batches

        loop.set_postfix({
            "loss": f"{mean_loss:.3f}",
            "box": f"{mean_box:.3f}",
            "obj": f"{mean_obj:.3f}",
            "noobj": f"{mean_noobj:.3f}",
            "cls": f"{mean_class:.3f}",
        })

    mean_loss = running_loss / num_batches
    mean_box = running_box / num_batches
    mean_obj = running_obj / num_batches
    mean_noobj = running_noobj / num_batches
    mean_class = running_class / num_batches

    return (
        mean_loss,
        mean_box,
        mean_obj,
        mean_noobj,
        mean_class,
    )

def print_simple_table(metrics):
    header = f"{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}"
    separator = "-" * len(header)
    
    print(f"\n[ SUMMARY METRICS ]")
    print(f"mAP: {metrics['map']:.4f}")
    print(separator)
    print(header)
    print(separator)
    
    for class_id, m in metrics['per_class_metric'].items():
        print(f"{config.PASCAL_CLASSES[class_id]:<10} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f}")
        
    print(separator)
    o = metrics['overall']
    print(f"{'OVERALL':<10} | {o['precision']:<10.4f} | {o['recall']:<10.4f} | {o['f1']:<10.4f}")
    print(separator)


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"Total trainable parameters: {total_trainable_params}")
    train_loader, test_loader, _, _ = get_train_test_loader()


    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
    
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * config.NUM_EPOCHS,
        eta_min=1e-6
    )

    start_epoch = 0
    if config.LOAD_MODEL and os.path.exists(config.CHECKPOINT_FILE):
        start_epoch = load_checkpoint(
            config.CHECKPOINT_FILE,
            model,
            optimizer,
            scheduler,
            scaler
        )


    all_pred_boxes, all_true_boxes = get_evaluation_bboxes(
        test_loader, model, iou_threshold=config.IOU_THRESH, anchors=config.ANCHORS, threshold=config.MAP_CONF_THRESH
    )
    print(f"Number of pred boxes: {len(all_pred_boxes)}\nNumber of gt boxes: {len(all_true_boxes)}")
    print(all_pred_boxes[0], all_pred_boxes[1])
    # map = mean_average_precision(all_true_boxes, all_true_boxes)
    # print("MAP:",map.item())
    metrics = compute_metrics(all_pred_boxes, all_true_boxes)
    print_simple_table(metrics)
    # print(f"Starting from epoch {start_epoch}")


    # best_test_loss = float('inf')

    # train_log_file = "training_log.txt"


    # train_losses, test_loss = [], []

    # for epoch in range(start_epoch, config.NUM_EPOCHS):
    #     mean_loss, mean_box, mean_obj, mean_noobj, mean_class = train_fn(
    #         train_loader,
    #         model,
    #         optimizer,
    #         loss_fn,
    #         scaler,
    #         epoch,
    #         scheduler,
    #     )

    #     if config.SAVE_MODEL:
    #         save_checkpoint(
    #             model,
    #             optimizer,
    #             scheduler,
    #             scaler,
    #             epoch,
    #             config.CHECKPOINT_FILE
    #         )

    #     if epoch + 1 == 12:
    #         print("UNFREEZING BACKBONE!")
    #         for param in model.backbone.parameters():
    #             param.requires_grad = True

    #     if epoch > 0 and (epoch+1) % 3 == 0:
    #         val_loss, val_box, val_obj, val_noobj, val_class = eval_fn(
    #             test_loader,
    #             model,
    #             loss_fn,
    #         )
    #         with open(train_log_file, "a") as f:
    #             f.write(f"{epoch+1}, {mean_loss:.4f}, {val_loss:.4f}\n")

    #         if val_loss < best_test_loss:
    #             best_test_loss = val_loss
    #             save_model(model, config.BEST_WEIGHTS_FILE)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
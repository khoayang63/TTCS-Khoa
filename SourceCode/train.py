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
from eval import *

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

    warmup_epochs = 5
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
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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



def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

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



    print(f"Starting from epoch {start_epoch}")

    # Tự động unfreeze nếu load checkpoint từ epoch 12 trở đi
    if start_epoch >= 12:
        print("Starting from epoch >= 12: UNFREEZING BACKBONE initially!")
        for param in model.backbone.parameters():
            param.requires_grad = True

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"Total trainable parameters: {total_trainable_params}")
    # if any(not param.requires_grad for param in model.backbone.parameters()):
    #     print("Backbone is frozen")
    # else:
    #     print("Backbone is NOT frozen")
    best_test_loss = float('inf')

    train_log_file = "training_log2.txt"


    train_losses, test_loss = [], []

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        mean_loss, mean_box, mean_obj, mean_noobj, mean_class = train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler,
            epoch,
            scheduler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                config.CHECKPOINT_FILE
            )

        if epoch + 1 == 12:
            print("UNFREEZING BACKBONE!")
            for param in model.backbone.parameters():
                param.requires_grad = True

        if epoch > 0 and (epoch+1) % 3 == 0:
            val_loss, val_box, val_obj, val_noobj, val_class = eval_fn(
                test_loader,
                model,
                loss_fn,
            )
            
            # Ghi nhật ký chi tiết tất cả các loại loss
            if not os.path.exists(train_log_file):
                with open(train_log_file, "w") as f:
                    f.write("Epoch,T_Loss,T_Box,T_Obj,T_NoObj,T_Class,V_Loss,V_Box,V_Obj,V_NoObj,V_Class\n")
            
            with open(train_log_file, "a") as f:
                f.write(f"{epoch+1},{mean_loss:.4f},{mean_box:.4f},{mean_obj:.4f},{mean_noobj:.4f},{mean_class:.4f},"
                        f"{val_loss:.4f},{val_box:.4f},{val_obj:.4f},{val_noobj:.4f},{val_class:.4f}\n")

            if val_loss < best_test_loss:
                best_test_loss = val_loss
                save_model(model, config.BEST_WEIGHTS_FILE)
        
        if (epoch+1) % 20 == 0:
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
    import multiprocessing
    multiprocessing.freeze_support()
    main()
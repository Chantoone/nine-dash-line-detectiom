"""
Engine huấn luyện và đánh giá mô hình.
"""
import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision


def train_one_epoch(model, optimizer, data_loader, device, epoch, kd_loss_fn):
    """
    Huấn luyện mô hình trong một epoch.
    
    Args:
        model: Mô hình EnhancedFasterRCNN.
        optimizer: Trình tối ưu hóa.
        data_loader: DataLoader cho dữ liệu huấn luyện.
        device: Thiết bị (cuda/cpu).
        epoch: Số epoch hiện tại (0-indexed).
        kd_loss_fn: Hàm mất mát Knowledge Distillation.
    
    Returns:
        Dict chứa average losses.
    """
    model.train()
    kd_loss_fn.to(device)
    total_main_loss = 0
    total_kd_loss = 0
    
    loop = tqdm(data_loader, desc=f"Epoch {epoch+1} Training")
    for images, targets in loop:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict, features, proposals, detections, image_sizes = model(images, targets)
        
        main_losses = sum(loss for loss in loss_dict.values())
        kd_loss = kd_loss_fn(features, proposals, targets, image_sizes)
        total_loss = main_losses + kd_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loop.set_postfix(
            main_loss=main_losses.item(), 
            kd_loss=kd_loss.item(), 
            total_loss=total_loss.item()
        )
                  
        total_main_loss += main_losses.item()
        if isinstance(kd_loss, torch.Tensor):
            total_kd_loss += kd_loss.item()
        
    avg_main_loss = total_main_loss / len(data_loader)
    avg_kd_loss = total_kd_loss / len(data_loader)
    
    return {
        "main_loss": avg_main_loss,
        "kd_loss": avg_kd_loss,
        "total_loss": avg_main_loss + avg_kd_loss,
    }


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Đánh giá mô hình trên tập validation.
    
    Args:
        model: Mô hình.
        data_loader: DataLoader cho dữ liệu validation.
        device: Thiết bị.
    
    Returns:
        Dict chứa kết quả đánh giá (mAP, mAR, ...).
    """
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy')
    metric.to(device)

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predictions = model(images)
        metric.update(predictions, targets)

    results = metric.compute()
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in results.items()}

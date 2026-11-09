#!/usr/bin/env python3
"""
Script huấn luyện chính cho Nine Dash Line Detection.
Sử dụng: python train.py [--config CONFIG_PATH]
"""
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.default import Config
from src.data.dataset import NineDashLineDataset, get_transform, collate_fn
from src.models.faster_rcnn import get_model
from src.models.kd_loss import KnowledgeDistillationLoss
from src.engine.trainer import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train Nine Dash Line Detector")
    parser.add_argument("--train-img-dir", type=str, default=None, help="Đường dẫn ảnh train")
    parser.add_argument("--train-ann-file", type=str, default=None, help="File annotation train")
    parser.add_argument("--valid-img-dir", type=str, default=None, help="Đường dẫn ảnh validation")
    parser.add_argument("--valid-ann-file", type=str, default=None, help="File annotation validation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    
    # Override config with command-line args
    if args.train_img_dir:
        config.data.train_img_dir = args.train_img_dir
    if args.train_ann_file:
        config.data.train_ann_file = args.train_ann_file
    if args.valid_img_dir:
        config.data.valid_img_dir = args.valid_img_dir
    if args.valid_ann_file:
        config.data.valid_ann_file = args.valid_ann_file
    if args.epochs:
        config.train.num_epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.checkpoint_dir:
        config.train.checkpoint_dir = args.checkpoint_dir

    device = config.get_device()
    print(f"Sử dụng thiết bị: {device}")

    # Kiểm tra dữ liệu
    if not os.path.exists(config.data.train_img_dir):
        print(f"LỖI: Không tìm thấy thư mục dữ liệu train: {config.data.train_img_dir}")
        return
    if not os.path.exists(config.data.valid_img_dir):
        print(f"LỖI: Không tìm thấy thư mục dữ liệu validation: {config.data.valid_img_dir}")
        return

    # Chuẩn bị dữ liệu
    print("Đang chuẩn bị dữ liệu...")
    dataset_train = NineDashLineDataset(
        root=config.data.train_img_dir,
        annFile=config.data.train_ann_file,
        transforms=get_transform(train=True)
    )
    dataset_valid = NineDashLineDataset(
        root=config.data.valid_img_dir,
        annFile=config.data.valid_ann_file,
        transforms=get_transform(train=False)
    )
    
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn
    )
    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Đã tải {len(dataset_train)} ảnh train và {len(dataset_valid)} ảnh validation.")

    # Khởi tạo mô hình
    print("Đang khởi tạo mô hình...")
    model = get_model(config.model.num_classes, config.model.pretrained)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, 
        lr=config.train.learning_rate, 
        momentum=config.train.momentum, 
        weight_decay=config.train.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.train.lr_step_size, 
        gamma=config.train.lr_gamma
    )

    # Knowledge Distillation Loss
    kd_loss_fn = KnowledgeDistillationLoss(
        beta1=config.train.kd_beta1, 
        beta2=config.train.kd_beta2, 
        iou_threshold=config.train.kd_iou_threshold
    )

    # Tạo thư mục checkpoint
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    
    # Huấn luyện
    print("\n" + "=" * 20 + " BẮT ĐẦU HUẤN LUYỆN " + "=" * 20)
    best_map = 0.0
    
    for epoch in range(config.train.num_epochs):
        # Train
        train_metrics = train_one_epoch(model, optimizer, data_loader_train, device, epoch, kd_loss_fn)
        print(f"--- Epoch {epoch+1} ---")
        print(f"  Main Loss: {train_metrics['main_loss']:.4f}")
        print(f"  KD Loss: {train_metrics['kd_loss']:.4f}")
        print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
        
        lr_scheduler.step()
        
        # Evaluate
        eval_results = evaluate(model, data_loader_valid, device)
        print(f"  mAP: {eval_results['map']:.4f}")
        print(f"  mAP@50: {eval_results['map_50']:.4f}")
        print(f"  AR@100: {eval_results['mar_100']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.train.save_every == 0:
            checkpoint_path = os.path.join(
                config.train.checkpoint_dir, f"model_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_results': eval_results,
            }, checkpoint_path)
            print(f"  Đã lưu checkpoint: {checkpoint_path}")
        
        # Save best model
        current_map = eval_results['map']
        if current_map > best_map:
            best_map = current_map
            best_path = os.path.join(config.train.checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'eval_results': eval_results,
            }, best_path)
            print(f"  ★ Mô hình tốt nhất mới! mAP={best_map:.4f}")

    print("\n" + "=" * 20 + " HOÀN THÀNH HUẤN LUYỆN " + "=" * 20)
    print(f"Best mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()

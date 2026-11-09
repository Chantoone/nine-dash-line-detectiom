"""
Cấu hình mặc định cho pipeline huấn luyện và inference.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """Cấu hình dữ liệu."""
    train_img_dir: str = "data/train/images"
    train_ann_file: str = "data/train/_annotations.coco.json"
    valid_img_dir: str = "data/valid/images"
    valid_ann_file: str = "data/valid/_annotations.coco.json"
    batch_size: int = 4
    num_workers: int = 2
    image_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, ...] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Cấu hình mô hình."""
    num_classes: int = 2  # background + nine_dash_line
    pretrained: bool = True
    confidence_threshold: float = 0.5


@dataclass
class TrainConfig:
    """Cấu hình huấn luyện."""
    num_epochs: int = 20
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 5
    lr_gamma: float = 0.1
    # Knowledge Distillation
    kd_beta1: float = 1.0
    kd_beta2: float = 0.5
    kd_iou_threshold: float = 0.5
    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1


@dataclass
class Config:
    """Cấu hình tổng hợp."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    def get_device(self):
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

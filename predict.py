#!/usr/bin/env python3
"""
Script inference cho Nine Dash Line Detection.
Sử dụng: python predict.py --image IMAGE_PATH --checkpoint CHECKPOINT_PATH [--threshold 0.5]
"""
import argparse
import cv2
import torch
import numpy as np
from PIL import Image

from configs.default import Config
from src.models.faster_rcnn import get_model
from src.data.dataset import get_transform
from src.utils.visualization import draw_detections, denormalize_image


def predict(image_path: str, checkpoint_path: str, threshold: float = 0.5, device: str = "auto"):
    """
    Thực hiện inference trên một ảnh.
    
    Args:
        image_path: Đường dẫn ảnh đầu vào.
        checkpoint_path: Đường dẫn file checkpoint (.pth).
        threshold: Ngưỡng confidence.
        device: Thiết bị sử dụng.
    
    Returns:
        Tuple (ảnh gốc, predictions dict).
    """
    config = Config()
    config.device = device
    dev = config.get_device()
    
    # Load model
    model = get_model(config.model.num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(dev)
    model.eval()
    
    # Load và tiền xử lý ảnh
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    
    transform = get_transform(train=False)
    transformed = transform(image=np.array(image), bboxes=[], labels=[])
    image_tensor = transformed['image'].unsqueeze(0).to(dev)
    
    # Inference
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # Lọc theo threshold
    keep = predictions['scores'] >= threshold
    filtered_predictions = {
        'boxes': predictions['boxes'][keep],
        'labels': predictions['labels'][keep],
        'scores': predictions['scores'][keep],
    }
    
    return original_image, filtered_predictions


def main():
    parser = argparse.ArgumentParser(description="Nine Dash Line Detection - Inference")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn ảnh")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng confidence")
    parser.add_argument("--output", type=str, default="output.jpg", help="Đường dẫn ảnh kết quả")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    print(f"Đang xử lý: {args.image}")
    original_image, predictions = predict(
        args.image, args.checkpoint, args.threshold, args.device
    )
    
    num_detections = len(predictions['boxes'])
    print(f"Phát hiện {num_detections} đối tượng (threshold={args.threshold})")
    
    result_image = draw_detections(original_image, predictions, args.threshold)
    cv2.imwrite(args.output, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print(f"Đã lưu kết quả: {args.output}")


if __name__ == '__main__':
    main()

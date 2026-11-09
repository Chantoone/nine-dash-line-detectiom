"""
Tiện ích trực quan hóa kết quả phát hiện.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple


# Màu sắc cho bounding box
COLORS = {
    "nine_dash_line": (0, 0, 255),  # Đỏ (BGR)
    "default": (0, 255, 0),          # Xanh lá
}

CLASS_NAMES = {
    0: "background",
    1: "nine_dash_line",
}


def draw_detections(
    image: np.ndarray,
    predictions: Dict,
    confidence_threshold: float = 0.5,
    class_names: Dict = None,
) -> np.ndarray:
    """
    Vẽ bounding boxes lên ảnh.
    
    Args:
        image: Ảnh gốc (numpy array, RGB hoặc BGR).
        predictions: Dict chứa 'boxes', 'labels', 'scores'.
        confidence_threshold: Ngưỡng confidence để hiển thị.
        class_names: Mapping từ class id sang tên class.
    
    Returns:
        Ảnh đã vẽ bounding boxes.
    """
    if class_names is None:
        class_names = CLASS_NAMES
        
    result = image.copy()
    
    boxes = predictions.get('boxes', [])
    labels = predictions.get('labels', [])
    scores = predictions.get('scores', [])
    
    for box, label, score in zip(boxes, labels, scores):
        if score < confidence_threshold:
            continue
            
        if hasattr(box, 'cpu'):
            box = box.cpu().numpy()
        if hasattr(label, 'item'):
            label = label.item()
        if hasattr(score, 'item'):
            score = score.item()
            
        x1, y1, x2, y2 = map(int, box)
        
        class_name = class_names.get(label, f"class_{label}")
        color = COLORS.get(class_name, COLORS["default"])
        
        # Vẽ bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ label
        label_text = f"{class_name}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(result, (x1, y1 - text_h - baseline - 5), 
                      (x1 + text_w, y1), color, -1)
        cv2.putText(result, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result


def denormalize_image(
    image_tensor, 
    mean: Tuple = (0.485, 0.456, 0.406), 
    std: Tuple = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Chuyển đổi tensor đã normalize về ảnh numpy RGB.
    """
    import torch
    if isinstance(image_tensor, torch.Tensor):
        img = image_tensor.cpu().clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img
    return image_tensor

"""
Mô hình Faster R-CNN cải tiến cho phát hiện Đường Lưỡi Bò.
Kiến trúc: Faster R-CNN (ResNet-50-FPN) với forward tùy chỉnh cho Knowledge Distillation.
"""
import torch
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class EnhancedFasterRCNN(FasterRCNN):
    """
    Lớp mô hình kế thừa từ FasterRCNN của torchvision.
    Tùy chỉnh phương thức forward để trả về các thông tin cần thiết
    cho việc tính toán Knowledge Distillation Loss.
    """
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)

        proposals, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses, features, proposals, detections, images.image_sizes
        
        return detections


def get_model(num_classes: int = 2, pretrained: bool = True) -> EnhancedFasterRCNN:
    """
    Khởi tạo mô hình EnhancedFasterRCNN với backbone ResNet-50-FPN.
    
    Args:
        num_classes: Số lượng class (bao gồm background). Mặc định = 2.
        pretrained: Sử dụng pretrained weights. Mặc định = True.
    
    Returns:
        Mô hình EnhancedFasterRCNN đã được cấu hình.
    """
    weights = 'DEFAULT' if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.__class__ = EnhancedFasterRCNN
    return model

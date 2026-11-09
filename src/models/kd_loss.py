"""
Knowledge Distillation Loss cho Object Detection.
So sánh đặc trưng của các vùng đề xuất (proposals) với các vùng ground-truth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, MultiScaleRoIAlign


class KnowledgeDistillationLoss(nn.Module):
    """
    Tính toán KD Loss dựa trên việc so sánh đặc trưng của các VÙNG ĐỀ XUẤT (proposals)
    với đặc trưng của các VÙNG ĐÚNG (ground-truth), được trích xuất từ backbone features.
    
    Công thức: L_kd = β1 * L1_loss + β2 * L_irm
    """
    def __init__(self, beta1: float = 1.0, beta2: float = 0.1, iou_threshold: float = 0.3):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.iou_threshold = iou_threshold
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'], 
            output_size=7, 
            sampling_ratio=2
        )

    def _match_proposals_to_ground_truth(self, proposals, gt_boxes, iou_threshold):
        """Matching proposals với ground-truth boxes dựa trên IoU."""
        if proposals.shape[0] == 0 or gt_boxes.shape[0] == 0:
            return [], []
        iou_matrix = box_iou(proposals, gt_boxes)
        max_iou_per_gt, max_iou_indices_per_gt = iou_matrix.max(dim=0)
        matched_proposal_indices = []
        matched_gt_indices = []
        for i, iou in enumerate(max_iou_per_gt):
            if iou >= iou_threshold:
                proposal_idx = max_iou_indices_per_gt[i].item()
                if proposal_idx not in matched_proposal_indices:
                    matched_proposal_indices.append(proposal_idx)
                    matched_gt_indices.append(i)
        return matched_proposal_indices, matched_gt_indices

    def forward(self, features, proposals, targets, image_sizes):
        all_gt_embeddings = []
        all_pred_embeddings = []

        for i, (target, proposal) in enumerate(zip(targets, proposals)):
            gt_boxes = target['boxes']
            matched_proposal_indices, matched_gt_indices = self._match_proposals_to_ground_truth(
                proposal, gt_boxes, self.iou_threshold
            )
            if not matched_proposal_indices:
                continue

            matched_proposals = proposal[matched_proposal_indices]
            matched_gt_boxes = gt_boxes[matched_gt_indices]

            pred_aligned_features = self.roi_align(features, [matched_proposals], [image_sizes[i]])
            gt_aligned_features = self.roi_align(features, [matched_gt_boxes], [image_sizes[i]])

            pred_embeddings = F.adaptive_avg_pool2d(pred_aligned_features, (1, 1)).flatten(start_dim=1)
            gt_embeddings = F.adaptive_avg_pool2d(gt_aligned_features, (1, 1)).flatten(start_dim=1)
            
            pred_embeddings = F.normalize(pred_embeddings, p=2, dim=1)
            gt_embeddings = F.normalize(gt_embeddings, p=2, dim=1)
            
            all_pred_embeddings.append(pred_embeddings)
            all_gt_embeddings.append(gt_embeddings)

        if not all_gt_embeddings:
            return torch.tensor(0.0, device=proposals[0].device)

        final_gt_embeddings = torch.cat(all_gt_embeddings, dim=0)
        final_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)

        loss_l1 = F.l1_loss(final_pred_embeddings, final_gt_embeddings)
        SR = torch.matmul(final_pred_embeddings, final_pred_embeddings.T)
        SI = torch.matmul(final_gt_embeddings, final_gt_embeddings.T)
        loss_irm = torch.norm(SR - SI, p='fro')

        return self.beta1 * loss_l1 + self.beta2 * loss_irm

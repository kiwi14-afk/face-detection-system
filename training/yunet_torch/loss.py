"""
Loss functions for YuNet training.
- Focal Loss for classification
- EIoU Loss for bounding box regression
- SmoothL1 Loss for keypoints
- SimOTA-style target assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import bbox_decode, generate_priors


# ============================================================
#  EIoU Loss (Extended IoU) — from the YuNet paper
# ============================================================
class EIoULoss(nn.Module):
    """
    EIoU = IoU - (ρ²(b,b_gt)/c²) - (ρ²(w,w_gt)/Cw²) - (ρ²(h,h_gt)/Ch²)
    Loss = 1 - EIoU

    Where: ρ = Euclidean distance, c = diagonal of smallest enclosing box,
           Cw, Ch = width/height of smallest enclosing box.
    """

    def __init__(self, reduction: str = 'sum', loss_weight: float = 5.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] in (tl_x, tl_y, br_x, br_y) format
            target_boxes: [N, 4] in (tl_x, tl_y, br_x, br_y) format
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0

        # Convert to (cx, cy, w, h)
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]

        gt_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        gt_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        gt_w = target_boxes[:, 2] - target_boxes[:, 0]
        gt_h = target_boxes[:, 3] - target_boxes[:, 1]

        # IoU
        iou = _compute_iou(pred_boxes, target_boxes)

        # Center distance
        rho2_c = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

        # Smallest enclosing box
        enclose_lt_x = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_lt_y = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_rb_x = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_rb_y = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        enclose_w = enclose_rb_x - enclose_lt_x
        enclose_h = enclose_rb_y - enclose_lt_y
        c2 = enclose_w ** 2 + enclose_h ** 2 + 1e-6

        # Width/height consistency
        rho2_w = (pred_w - gt_w) ** 2
        rho2_h = (pred_h - gt_h) ** 2
        cw2 = enclose_w ** 2 + 1e-6
        ch2 = enclose_h ** 2 + 1e-6

        eiou = iou - rho2_c / c2 - rho2_w / cw2 - rho2_h / ch2
        loss = self.loss_weight * (1 - eiou)

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss


def _compute_iou(boxes1, boxes2):
    """Pairwise IoU for aligned boxes (N,4) vs (N,4)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter + 1e-6
    return inter / union


# ============================================================
#  Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss for binary classification with sigmoid."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] logits (before sigmoid)
            target: [N, C] one-hot targets
        """
        if pred.numel() == 0:
            return pred.sum() * 0.0

        pred_sigmoid = pred.sigmoid()
        # BCE with focal weighting
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma

        loss = alpha_factor * modulating_factor * ce_loss

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss


# ============================================================
#  Target Assignment (SimOTA-style, simplified)
# ============================================================
class SimOTAAssigner:
    """Simplified SimOTA target assignment for anchor-free detection."""

    def __init__(self, center_radius: float = 2.5):
        self.center_radius = center_radius

    @torch.no_grad()
    def assign(self, scores, priors, decoded_bboxes, gt_bboxes):
        """
        Assign targets using center-prior and IoU matching.

        Args:
            scores: [N, 1] combined classification+objectness scores
            priors: [N, 4] anchor priors (cx, cy, stride_w, stride_h)
            decoded_bboxes: [N, 4] predicted boxes (tl_x, tl_y, br_x, br_y)
            gt_bboxes: [M, 4] ground truth boxes

        Returns:
            pos_inds: indices of positive priors
            pos_gt_inds: corresponding GT indices for each positive prior
            ious: max IoU for each positive prior with its assigned GT
        """
        if len(gt_bboxes) == 0:
            return torch.zeros(0, dtype=torch.long, device=priors.device), \
                   torch.zeros(0, dtype=torch.long, device=priors.device), \
                   torch.zeros(0, device=priors.device)

        # Compute IoU between all priors and all GTs
        ious = self._compute_iou_matrix(decoded_bboxes, gt_bboxes)  # [N, M]

        # For each GT, find center priors (within center_radius * stride)
        gt_centers = torch.stack([
            (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2,
            (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2,
        ], dim=-1)  # [M, 2]

        center_mask = torch.zeros(len(priors), len(gt_bboxes), dtype=torch.bool, device=priors.device)
        for gt_idx in range(len(gt_bboxes)):
            dist = torch.sqrt(
                (priors[:, 0] - gt_centers[gt_idx, 0]) ** 2 +
                (priors[:, 1] - gt_centers[gt_idx, 1]) ** 2
            )
            center_mask[:, gt_idx] = dist <= (self.center_radius * priors[:, 2])

        # Combine: only consider center priors for each GT
        masked_ious = ious.clone()
        masked_ious[~center_mask] = -1.0

        # For each GT, pick top-k priors by IoU
        max_iou, max_idx = masked_ious.max(dim=0)  # [M]
        pos_mask = max_iou > 0.0
        pos_gt_inds = torch.where(pos_mask)[0]
        pos_inds = max_idx[pos_mask]

        # Get the matched IoUs
        matched_ious = max_iou[pos_mask]

        return pos_inds, pos_gt_inds, matched_ious

    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix: [N, M]."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2[None, :] - inter + 1e-6
        return inter / union


# ============================================================
#  Combined YuNet Loss
# ============================================================
class YuNetLoss(nn.Module):
    """Combined loss for YuNet training."""

    def __init__(
        self,
        num_classes: int = 1,
        strides=None,
        use_kps: bool = True,
        kps_num: int = 5,
    ):
        super().__init__()
        if strides is None:
            strides = [8, 16, 32]

        self.num_classes = num_classes
        self.strides = strides
        self.use_kps = use_kps
        self.NK = kps_num

        self.loss_cls = FocalLoss(alpha=0.25, gamma=2.0, reduction='sum')
        self.loss_bbox = EIoULoss(reduction='sum', loss_weight=5.0)
        self.loss_obj = FocalLoss(alpha=0.25, gamma=2.0, reduction='sum')
        if use_kps:
            self.loss_kps = nn.SmoothL1Loss(beta=1.0 / 9.0, reduction='sum')

        self.assigner = SimOTAAssigner(center_radius=2.5)

    def forward(self, predictions, gt_bboxes, gt_kpss, images):
        """
        Args:
            predictions: (cls_scores, bbox_preds, obj_preds, kps_preds) from YuNetHead
            gt_bboxes: list of [Mi, 4] tensors, one per image
            gt_kpss: list of [Mi, 5, 3] tensors (x, y, visibility) per image
            images: the original batch [B, 3, H, W] (not used directly, for context)
        """
        cls_scores, bbox_preds, obj_preds, kps_preds = predictions
        B = cls_scores[0].shape[0]
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype

        # Flatten predictions across all levels
        featmap_sizes = [(cs.shape[2], cs.shape[3]) for cs in cls_scores]

        flat_cls = []
        flat_bbox = []
        flat_obj = []
        flat_kps = []

        for level_idx in range(len(cls_scores)):
            flat_cls.append(cls_scores[level_idx].permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            flat_bbox.append(bbox_preds[level_idx].permute(0, 2, 3, 1).reshape(B, -1, 4))
            flat_obj.append(obj_preds[level_idx].permute(0, 2, 3, 1).reshape(B, -1, 1))
            if self.use_kps and kps_preds is not None:
                flat_kps.append(kps_preds[level_idx].permute(0, 2, 3, 1).reshape(B, -1, self.NK * 2))

        flat_cls = torch.cat(flat_cls, dim=1)     # [B, N_total, C]
        flat_bbox_raw = torch.cat(flat_bbox, dim=1)  # [B, N_total, 4]
        flat_obj = torch.cat(flat_obj, dim=1)     # [B, N_total, 1]
        if self.use_kps and kps_preds is not None:
            flat_kps = torch.cat(flat_kps, dim=1)  # [B, N_total, K*2]

        # Generate priors once (same for all images in batch)
        priors = generate_priors(featmap_sizes, self.strides, dtype=dtype, device=device)
        priors = priors.unsqueeze(0).repeat(B, 1, 1)  # [B, N_total, 4]

        # Decode bboxes
        flat_bboxes = bbox_decode(priors, flat_bbox_raw)  # [B, N_total, 4]

        total_loss_cls = torch.tensor(0.0, device=device)
        total_loss_bbox = torch.tensor(0.0, device=device)
        total_loss_obj = torch.tensor(0.0, device=device)
        total_loss_kps = torch.tensor(0.0, device=device)
        total_pos = 0

        for b in range(B):
            num_gts = len(gt_bboxes[b])
            if num_gts == 0:
                # No GT → all objectness targets are 0
                obj_target = torch.zeros(flat_obj.shape[1], 1, dtype=dtype, device=device)
                total_loss_obj += self.loss_obj(flat_obj[b], obj_target)
                continue

            gt_box = gt_bboxes[b].to(dtype=dtype, device=device)
            gt_kps = gt_kpss[b].to(dtype=dtype, device=device) if gt_kpss[b] is not None else None

            # Compute scores for assignment
            scores = flat_cls[b].sigmoid() * flat_obj[b].sigmoid()

            # Assign targets
            pos_inds, pos_gt_inds, pos_ious = self.assigner.assign(
                scores.squeeze(-1),
                priors[b],
                flat_bboxes[b],
                gt_box,
            )

            num_pos = len(pos_inds)
            if num_pos == 0:
                obj_target = torch.zeros(flat_obj.shape[1], 1, dtype=dtype, device=device)
                total_loss_obj += self.loss_obj(flat_obj[b], obj_target)
                continue

            total_pos += num_pos

            # === Classification targets ===
            cls_target = torch.zeros(num_pos, self.num_classes, dtype=dtype, device=device)
            cls_target.scatter_(1, torch.zeros(num_pos, 1, dtype=torch.long, device=device), 1)
            # Weight by IoU
            cls_target = cls_target * pos_ious.unsqueeze(-1)

            # === Objectness targets ===
            obj_target = torch.zeros(flat_obj.shape[1], 1, dtype=dtype, device=device)
            obj_target[pos_inds] = 1.0  # Simplified: binary target

            # === BBox targets ===
            bbox_target = gt_box[pos_gt_inds]  # [num_pos, 4]
            pos_pred_bbox = flat_bboxes[b][pos_inds]  # [num_pos, 4]

            # === KPS targets ===
            if self.use_kps and gt_kps is not None and kps_preds is not None:
                kps_target = gt_kps[pos_gt_inds, :, :2].reshape(-1, self.NK * 2)  # [num_pos, 10]
                kps_weight = gt_kps[pos_gt_inds, :, 2].mean(dim=1, keepdim=True)  # [num_pos, 1]
                pos_pred_kps = flat_kps[b][pos_inds]  # [num_pos, 10]

            # === Compute losses ===
            total_loss_cls += self.loss_cls(flat_cls[b][pos_inds], cls_target)
            total_loss_bbox += self.loss_bbox(pos_pred_bbox, bbox_target)
            total_loss_obj += self.loss_obj(flat_obj[b], obj_target)
            if self.use_kps and gt_kps is not None and kps_preds is not None:
                eps = 1e-6
                kps_weight_sum = kps_weight.sum() + eps
                total_loss_kps += self.loss_kps(
                    pos_pred_kps * kps_weight,
                    kps_target * kps_weight,
                )

        # Normalize by number of positive samples (like the original: reduce_mean)
        num_total = max(total_pos, 1)
        loss_dict = {
            'loss_cls': total_loss_cls / num_total,
            'loss_bbox': total_loss_bbox / num_total,
            'loss_obj': total_loss_obj / num_total,
            'loss_kps': total_loss_kps / num_total if self.use_kps else torch.tensor(0.0, device=device),
            'num_pos': total_pos,
        }
        loss_dict['total'] = sum(loss_dict[k] for k in loss_dict if k.startswith('loss_'))

        return loss_dict

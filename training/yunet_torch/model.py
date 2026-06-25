"""
YuNet: A Tiny Millisecond-level Face Detector
Pure PyTorch reimplementation — exact architecture from official MMDetection config.

Paper: https://link.springer.com/article/10.1007/s11633-023-1423-y
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  Building Blocks
# ============================================================
class ConvDPUnit(nn.Module):
    """Depthwise-Pointwise Convolution Unit.
    1x1 conv → 3x3 depthwise conv → BN → ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, with_bn_relu: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels)
        self.with_bn_relu = with_bn_relu
        if with_bn_relu:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.with_bn_relu:
            x = self.bn(x)
            x = self.relu(x)
        return x


class ConvHead(nn.Module):
    """Stem block: 3x3 conv(s=2) → BN → ReLU → ConvDPUnit"""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ConvDPUnit(mid_ch, out_ch, True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class Conv4LayerBlock(nn.Module):
    """Two ConvDPUnits stacked."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvDPUnit(in_ch, in_ch, True)
        self.conv2 = ConvDPUnit(in_ch, out_ch, True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ============================================================
#  Backbone
# ============================================================
class YuNetBackbone(nn.Module):
    """
    stage_channels: [[3,16,16], [16,64], [64,64], [64,64], [64,64], [64,64]]
    downsample_idx: [0, 2, 3, 4]
    out_idx:        [3, 4, 5]
    Output strides: [8, 16, 32]
    """

    def __init__(
        self,
        stage_channels=None,
        downsample_idx=None,
        out_idx=None,
    ):
        super().__init__()
        if stage_channels is None:
            stage_channels = [[3, 16, 16], [16, 64], [64, 64], [64, 64], [64, 64], [64, 64]]
        if downsample_idx is None:
            downsample_idx = [0, 2, 3, 4]
        if out_idx is None:
            out_idx = [3, 4, 5]

        self.layer_num = len(stage_channels)
        self.downsample_idx = downsample_idx
        self.out_idx = out_idx

        # Stage 0: Stem
        self.model0 = ConvHead(*stage_channels[0])
        # Stages 1-5
        for i in range(1, self.layer_num):
            self.add_module(f'model{i}', Conv4LayerBlock(*stage_channels[i]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        outs = []
        for i in range(self.layer_num):
            x = getattr(self, f'model{i}')(x)
            if i in self.out_idx:
                outs.append(x)
            if i in self.downsample_idx:
                x = F.max_pool2d(x, 2)
        return outs


# ============================================================
#  Tiny FPN (Neck)
# ============================================================
class TFPN(nn.Module):
    """Tiny Feature Pyramid Network — top-down fusion with nearest-neighbor upsample."""

    def __init__(self, in_channels=None, out_idx=None):
        super().__init__()
        if in_channels is None:
            in_channels = [64, 64, 64]
        if out_idx is None:
            out_idx = [0, 1, 2]

        self.num_layers = len(in_channels)
        self.out_idx = out_idx
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.lateral_convs.append(ConvDPUnit(in_channels[i], in_channels[i], True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        # Top-down flow
        for i in range(self.num_layers - 1, 0, -1):
            feats[i] = self.lateral_convs[i](feats[i])
            feats[i - 1] = feats[i - 1] + F.interpolate(feats[i], scale_factor=2.0, mode='nearest')

        feats[0] = self.lateral_convs[0](feats[0])
        outs = [feats[i] for i in self.out_idx]
        return outs


# ============================================================
#  Detection Head
# ============================================================
class YuNetHead(nn.Module):
    """
    Anchor-free detection head (YOLOX-style).
    Per feature level: shared conv → cls/bbox/obj/kps branches.
    Strides: [8, 16, 32]
    """

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 64,
        feat_channels: int = 64,
        stacked_convs: int = 0,
        shared_stacked_convs: int = 1,
        strides=None,
        use_kps: bool = True,
        kps_num: int = 5,
    ):
        super().__init__()
        if strides is None:
            strides = [8, 16, 32]

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.use_kps = use_kps
        self.NK = kps_num
        self.strides = strides
        self.strides_num = len(strides)

        # Shared convolutions per level
        if shared_stacked_convs > 0:
            self.share_convs = nn.ModuleList()
            for _ in strides:
                layers = []
                for j in range(shared_stacked_convs):
                    chn = in_channels if j == 0 else feat_channels
                    layers.append(ConvDPUnit(chn, feat_channels))
                self.share_convs.append(nn.Sequential(*layers))
        else:
            self.share_convs = None

        # Task-specific convolutions
        if stacked_convs > 0:
            self.cls_convs = nn.ModuleList()
            self.reg_convs = nn.ModuleList()
            for _ in strides:
                cls_layers, reg_layers = [], []
                for j in range(stacked_convs):
                    chn = in_channels if j == 0 and shared_stacked_convs == 0 else feat_channels
                    cls_layers.append(ConvDPUnit(chn, feat_channels))
                    reg_layers.append(ConvDPUnit(chn, feat_channels))
                self.cls_convs.append(nn.Sequential(*cls_layers))
                self.reg_convs.append(nn.Sequential(*reg_layers))
        else:
            self.cls_convs = None
            self.reg_convs = None

        # Prediction heads
        chn = feat_channels if (stacked_convs > 0 or shared_stacked_convs > 0) else in_channels
        self.cls_preds = nn.ModuleList([ConvDPUnit(chn, num_classes, False) for _ in strides])
        self.bbox_preds = nn.ModuleList([ConvDPUnit(chn, 4, False) for _ in strides])
        self.obj_preds = nn.ModuleList([ConvDPUnit(chn, 1, False) for _ in strides])
        if use_kps:
            self.kps_preds = nn.ModuleList([ConvDPUnit(chn, kps_num * 2, False) for _ in strides])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        """Returns: cls_scores, bbox_preds, obj_preds, [kps_preds]"""
        if self.share_convs is not None:
            feats = [conv(f) for f, conv in zip(feats, self.share_convs)]

        if self.cls_convs is not None:
            feats_cls = [conv(f) for f, conv in zip(feats, self.cls_convs)]
            feats_reg = [conv(f) for f, conv in zip(feats, self.reg_convs)]
        else:
            feats_cls = feats_reg = feats

        cls_scores = [conv(f) for f, conv in zip(feats_cls, self.cls_preds)]
        bbox_preds = [conv(f) for f, conv in zip(feats_reg, self.bbox_preds)]
        obj_preds = [conv(f) for f, conv in zip(feats_reg, self.obj_preds)]

        if self.use_kps:
            kps_preds = [conv(f) for f, conv in zip(feats_reg, self.kps_preds)]
            return cls_scores, bbox_preds, obj_preds, kps_preds

        return cls_scores, bbox_preds, obj_preds, None


# ============================================================
#  Full YuNet Model
# ============================================================
class YuNet(nn.Module):
    """Complete YuNet face detector."""

    def __init__(self, num_classes: int = 1, use_kps: bool = True):
        super().__init__()
        self.backbone = YuNetBackbone()
        self.neck = TFPN()
        self.head = YuNetHead(num_classes=num_classes, use_kps=use_kps)
        self.strides = self.head.strides

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)


# ============================================================
#  Prior Generator + Decode utilities
# ============================================================
def generate_priors(featmap_sizes, strides, dtype, device):
    """Generate anchor priors (cx, cy, stride_w, stride_h) for each feature level."""
    mlvl_priors = []
    for stride, (fh, fw) in zip(strides, featmap_sizes):
        shift_y, shift_x = torch.meshgrid(
            torch.arange(fh, dtype=dtype, device=device),
            torch.arange(fw, dtype=dtype, device=device),
            indexing='ij',
        )
        shifts = torch.stack([shift_x, shift_y], dim=-1)  # [fh, fw, 2]
        # cx, cy = (x + 0.5) * stride, (y + 0.5) * stride  → for decode we need grid centers
        # For priors (no offset): cx = x * stride, cy = y * stride
        cx = (shift_x + 0) * stride
        cy = (shift_y + 0) * stride
        sw = torch.full_like(cx, stride)
        sh = torch.full_like(cy, stride)
        priors = torch.stack([cx, cy, sw, sh], dim=-1).reshape(-1, 4)
        mlvl_priors.append(priors)
    return torch.cat(mlvl_priors, dim=0)


def bbox_decode(priors, bbox_preds):
    """Decode bbox predictions: xy = pred[:2] * stride + cx_cy, wh = exp(pred[2:]) * stride"""
    xys = bbox_preds[..., :2] * priors[..., 2:] + priors[..., :2]
    whs = bbox_preds[..., 2:].exp() * priors[..., 2:]
    tl_x = xys[..., 0] - whs[..., 0] / 2
    tl_y = xys[..., 1] - whs[..., 1] / 2
    br_x = xys[..., 0] + whs[..., 0] / 2
    br_y = xys[..., 1] + whs[..., 1] / 2
    return torch.stack([tl_x, tl_y, br_x, br_y], dim=-1)


def kps_decode(priors, kps_preds, num_kps=5):
    """Decode keypoint predictions."""
    decoded = []
    for i in range(num_kps):
        x = kps_preds[..., 2 * i] * priors[..., 2] + priors[..., 0]
        y = kps_preds[..., 2 * i + 1] * priors[..., 3] + priors[..., 1]
        decoded.extend([x, y])
    return torch.stack(decoded, dim=-1)


# ============================================================
#  Post-processing (for inference)
# ============================================================
def postprocess(cls_scores, bbox_preds, obj_preds, kps_preds,
                strides, score_thr=0.5, nms_thr=0.45, top_k=-1):
    """Convert raw network outputs to detection results (NMS applied)."""
    B = cls_scores[0].shape[0]
    all_results = []

    for b in range(B):
        # Collect predictions across all levels
        all_cls, all_bbox, all_obj, all_kps = [], [], [], []
        all_priors_list = []

        for level_idx, stride in enumerate(strides):
            _, _, h, w = cls_scores[level_idx].shape
            priors = generate_priors([(h, w)], [stride],
                                     dtype=cls_scores[level_idx].dtype,
                                     device=cls_scores[level_idx].device)
            all_priors_list.append(priors)

            cls_pred = cls_scores[level_idx][b].permute(1, 2, 0).reshape(-1, 1).sigmoid()
            obj_pred = obj_preds[level_idx][b].permute(1, 2, 0).reshape(-1, 1).sigmoid()
            bbox_pred = bbox_preds[level_idx][b].permute(1, 2, 0).reshape(-1, 4)
            decoded = bbox_decode(priors, bbox_pred)

            all_cls.append(cls_pred)
            all_bbox.append(decoded)
            all_obj.append(obj_pred)
            if kps_preds is not None:
                kps_pred = kps_preds[level_idx][b].permute(1, 2, 0).reshape(-1, 10)
                all_kps.append(kps_decode(priors, kps_pred, num_kps=5))
            else:
                all_kps.append(torch.zeros(priors.shape[0], 10, device=priors.device))

        all_cls = torch.cat(all_cls, dim=0)
        all_bbox = torch.cat(all_bbox, dim=0)
        all_obj = torch.cat(all_obj, dim=0)
        all_kps = torch.cat(all_kps, dim=0)

        scores = (all_cls * all_obj).squeeze(-1)
        valid = scores >= score_thr

        if valid.sum() == 0:
            all_results.append((torch.zeros(0, 5), torch.zeros(0, 10), torch.zeros(0)))
            continue

        keep_bbox = all_bbox[valid]
        keep_scores = scores[valid]
        keep_kps = all_kps[valid]

        # NMS
        keep = _nms(keep_bbox, keep_scores, nms_thr, top_k)
        all_results.append((keep_bbox[keep], keep_scores[keep], keep_kps[keep]))

    return all_results


def _nms(boxes, scores, iou_threshold, top_k):
    """Simple batched NMS."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Sort by score descending
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        if len(keep) >= top_k > 0:
            break
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # Compute IoU of the kept box with the rest
        box_i = boxes[i]
        other_boxes = boxes[order[1:]]

        ious = _box_iou(box_i.unsqueeze(0), other_boxes)[0]
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _box_iou(boxes1, boxes2):
    """Compute pairwise IoU between two sets of boxes (tl_x, tl_y, br_x, br_y)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou

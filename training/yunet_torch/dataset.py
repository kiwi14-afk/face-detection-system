"""
WIDER FACE dataset loader for YuNet training.
Supports both SCRFD labelv2 format and original WIDER FACE annotation format.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
#  Data augmentation transforms
# ============================================================
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes, keypoints):
        for t in self.transforms:
            image, bboxes, keypoints = t(image, bboxes, keypoints)
        return image, bboxes, keypoints


class RandomSquareCrop:
    """Random square crop with scale choice (from SCRFD)."""

    def __init__(self, crop_choice=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5)):
        self.crop_choice = crop_choice

    def __call__(self, image, bboxes, keypoints):
        h, w = image.shape[:2]
        scale = random.choice(self.crop_choice)
        crop_size = int(min(h, w) * scale)
        crop_size = min(crop_size, h, w)

        top = random.randint(0, max(0, h - crop_size))
        left = random.randint(0, max(0, w - crop_size))

        # Crop image
        image = image[top:top + crop_size, left:left + crop_size]

        # Adjust bboxes
        if len(bboxes) > 0:
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] -= left
            bboxes[:, [1, 3]] -= top
            # Clip to valid range
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, crop_size)
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, crop_size)

            # Remove invalid boxes
            valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
            bboxes = bboxes[valid]
            if len(keypoints) > 0:
                keypoints = keypoints[valid]

        # Adjust keypoints
        if len(keypoints) > 0:
            keypoints = keypoints.copy()
            keypoints[:, :, 0] -= left
            keypoints[:, :, 1] -= top

        return image, bboxes, keypoints


class Resize:
    """Resize image to target size."""

    def __init__(self, img_scale=(640, 640), keep_ratio=False):
        self.img_scale = tuple(img_scale)
        self.keep_ratio = keep_ratio

    def __call__(self, image, bboxes, keypoints):
        h, w = image.shape[:2]
        target_w, target_h = self.img_scale

        if self.keep_ratio:
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Pad to target size
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Adjust bboxes
            scale_x, scale_y = scale, scale
            offset_x, offset_y = left, top
        else:
            scale_x = target_w / w
            scale_y = target_h / h
            offset_x, offset_y = 0, 0
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Adjust bboxes
        if len(bboxes) > 0:
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_x + offset_x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_y + offset_y

        # Adjust keypoints
        if len(keypoints) > 0:
            keypoints = keypoints.copy()
            keypoints[:, :, 0] = keypoints[:, :, 0] * scale_x + offset_x
            keypoints[:, :, 1] = keypoints[:, :, 1] * scale_y + offset_y

        return image, bboxes, keypoints


class RandomFlip:
    """Random horizontal flip."""

    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, image, bboxes, keypoints):
        if random.random() < self.flip_ratio:
            h, w = image.shape[:2]
            image = cv2.flip(image, 1)

            if len(bboxes) > 0:
                bboxes = bboxes.copy()
                x1 = w - bboxes[:, 2]
                x2 = w - bboxes[:, 0]
                bboxes[:, 0] = x1
                bboxes[:, 2] = x2

            if len(keypoints) > 0:
                keypoints = keypoints.copy()
                keypoints[:, :, 0] = w - keypoints[:, :, 0]

        return image, bboxes, keypoints


# ============================================================
#  WIDER FACE Dataset
# ============================================================
class WiderFaceDataset(Dataset):
    """WIDER FACE dataset for training.

    Supports two annotation formats:
    1. SCRFD labelv2: each line = "# image_path bbox_count face_id x1 y1 w h blur illumination occlusion pose x1 y1 x2 y2 ..."
       Or simpler: "image_path x1 y1 w h ... xk yk"
    2. Original WIDER: wider_face_train_bbx_gt.txt
    """

    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        input_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        max_faces_per_image: int = 200,
    ):
        self.img_prefix = Path(img_prefix)
        self.input_size = input_size
        self.augment = augment
        self.max_faces = max_faces_per_image

        # Build augmentation pipeline
        if augment:
            self.transform = Compose([
                RandomSquareCrop(crop_choice=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
                Resize(img_scale=input_size, keep_ratio=False),
                RandomFlip(flip_ratio=0.5),
            ])
        else:
            self.transform = Compose([
                Resize(img_scale=input_size, keep_ratio=True),
            ])

        # Parse annotations
        self.samples = self._parse_annotations(ann_file)
        print(f"Loaded {len(self.samples)} images from {ann_file}")

    def _parse_annotations(self, ann_file: str):
        """Parse annotation file and build sample index."""
        samples = []

        with open(ann_file, 'r') as f:
            lines = f.readlines()

        # Try to detect format
        # SCRFD labelv2 format: first line often starts with '#'
        if lines[0].startswith('#'):
            return self._parse_labelv2(lines)
        else:
            return self._parse_wider_original(lines)

    def _parse_labelv2(self, lines):
        """Parse SCRFD labelv2 format."""
        samples = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            img_path = parts[0]
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Try without extension
                img_path_test = img_path + '.jpg'
                if (self.img_prefix / img_path_test).exists():
                    img_path = img_path_test

            num_faces = int(parts[1])
            offset = 2

            bboxes = []  # [x1, y1, w, h] → converted to [x1, y1, x2, y2]
            keypoints = []  # [5, 3] with (x, y, visibility) for each face
            face_ids = []

            for i in range(num_faces):
                if offset + 4 > len(parts):
                    break

                # Read bbox + attributes (if present)
                x1 = float(parts[offset])
                y1 = float(parts[offset + 1])
                w = float(parts[offset + 2])
                h = float(parts[offset + 3])
                offset += 4

                bboxes.append([x1, y1, x1 + w, y1 + h])

                # Check for challenge attributes (4 values: blur, illumination, occlusion, pose)
                attrs = {'blur': 0, 'illumination': 0, 'occlusion': 0, 'pose': 0}
                if offset + 4 <= len(parts):
                    # Might have attributes
                    potential_attrs = parts[offset:offset + 4]
                    all_int_or_float = all(
                        p.replace('.', '').replace('-', '').isdigit()
                        for p in potential_attrs
                    )
                    if all_int_or_float and all(float(p) <= 2.0 for p in potential_attrs):
                        # These are probably attribute values
                        attrs = {
                            'blur': int(float(parts[offset])),
                            'illumination': int(float(parts[offset + 1])),
                            'occlusion': int(float(parts[offset + 2])),
                            'pose': int(float(parts[offset + 3])),
                        }
                        offset += 4

                # Read keypoints (5 points, 3 values each = 15 values)
                kp = np.zeros((5, 3), dtype=np.float32)
                kp[:, 2] = 1.0  # Default visibility
                if offset + 15 <= len(parts):
                    for k in range(5):
                        if offset + 3 <= len(parts):
                            kp[k, 0] = float(parts[offset])
                            kp[k, 1] = float(parts[offset + 1])
                            kp[k, 2] = float(parts[offset + 2])
                            offset += 3
                keypoints.append(kp)
                face_ids.append(attrs)

            if len(bboxes) > 0:
                samples.append({
                    'img_path': str(self.img_prefix / img_path),
                    'bboxes': np.array(bboxes, dtype=np.float32),
                    'keypoints': np.array(keypoints, dtype=np.float32),
                    'attrs': face_ids,
                })

        return samples

    def _parse_wider_original(self, lines):
        """Parse original WIDER FACE annotation format.
        Format:
            <image_path>
            <num_faces>
            <x1> <y1> <w> <h> <blur> <expression> <illumination> <invalid> <occlusion> <pose>
            ...
        """
        samples = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            img_path = line
            i += 1

            if i >= len(lines):
                break

            num_faces = int(lines[i].strip())
            i += 1

            bboxes = []
            keypoints = []
            attrs_list = []
            for _ in range(num_faces):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                i += 1
                if len(parts) < 4:
                    continue

                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])

                bboxes.append([x, y, x + w, y + h])

                # Attributes (if available)
                attrs = {'blur': 0, 'illumination': 0, 'occlusion': 0, 'pose': 0}
                if len(parts) >= 10:
                    attrs['blur'] = int(parts[4])
                    attrs['illumination'] = int(parts[6])
                    attrs['occlusion'] = int(parts[8])
                    attrs['pose'] = int(parts[9])
                attrs_list.append(attrs)

                # WIDER original doesn't have keypoints, use zeros
                kp = np.zeros((5, 3), dtype=np.float32)
                kp[:, 2] = 1.0
                h_img = w * 1.2  # Estimate image height (not accurate, will be updated)
                # Set default keypoints at face center
                cx, cy = x + w / 2, y + h / 2
                kp[0] = [cx - w * 0.1, cy - h * 0.15, 1.0]  # left eye
                kp[1] = [cx + w * 0.1, cy - h * 0.15, 1.0]  # right eye
                kp[2] = [cx, cy, 1.0]  # nose
                kp[3] = [cx - w * 0.15, cy + h * 0.15, 1.0]  # left mouth
                kp[4] = [cx + w * 0.15, cy + h * 0.15, 1.0]  # right mouth
                keypoints.append(kp)

            if len(bboxes) > 0:
                samples.append({
                    'img_path': str(self.img_prefix / img_path),
                    'bboxes': np.array(bboxes, dtype=np.float32),
                    'keypoints': np.array(keypoints, dtype=np.float32),
                    'attrs': attrs_list,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Read image — always use imdecode to handle Chinese/Unicode paths on Windows
        img = cv2.imdecode(
            np.fromfile(sample['img_path'], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {sample['img_path']}")

        bboxes = sample['bboxes'].copy()
        keypoints = sample['keypoints'].copy()

        # Apply augmentations
        img, bboxes, keypoints = self.transform(img, bboxes, keypoints)

        # Filter out invalid boxes after augment
        if len(bboxes) > 0:
            valid_boxes = (bboxes[:, 2] > bboxes[:, 0] + 1) & (bboxes[:, 3] > bboxes[:, 1] + 1)
            bboxes = bboxes[valid_boxes]
            keypoints = keypoints[valid_boxes]

            # Limit max faces
            if len(bboxes) > self.max_faces:
                bboxes = bboxes[:self.max_faces]
                keypoints = keypoints[:self.max_faces]

        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]
        bboxes_tensor = torch.from_numpy(bboxes).float() if len(bboxes) > 0 else torch.zeros(0, 4)
        kps_tensor = torch.from_numpy(keypoints).float() if len(keypoints) > 0 else torch.zeros(0, 5, 3)

        return img_tensor, bboxes_tensor, kps_tensor


def collate_fn(batch):
    """Custom collate to handle variable number of GT boxes."""
    images = torch.stack([item[0] for item in batch], dim=0)
    gt_bboxes = [item[1] for item in batch]
    gt_kpss = [item[2] for item in batch]
    return images, gt_bboxes, gt_kpss


def create_dataloader(
    ann_file: str,
    img_prefix: str,
    batch_size: int = 16,
    input_size: Tuple[int, int] = (640, 640),
    augment: bool = True,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """Create a DataLoader for WIDER FACE."""
    dataset = WiderFaceDataset(
        ann_file=ann_file,
        img_prefix=img_prefix,
        input_size=input_size,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

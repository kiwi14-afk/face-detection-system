"""
YuNet Training Script
=====================
Train a YuNet face detector on WIDER FACE dataset.
Supports training on both original and preprocessed data.

Usage:
    python -m yunet_torch.train --data_dir data/WIDER_train --ann_file data/labelv2/train/labelv2.txt
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yunet_torch.model import YuNet
from yunet_torch.loss import YuNetLoss
from yunet_torch.dataset import WiderFaceDataset, create_dataloader, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train YuNet face detector")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--ann_file', type=str, required=True,
                        help='Path to annotation file (SCRFD labelv2 or WIDER original format)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=640,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--input_size', type=int, nargs=2, default=[640, 640],
                        help='Input image size (width height)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--val_ann', type=str, default=None,
                        help='Validation annotation file')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation image directory')
    parser.add_argument('--use_kps', action='store_true', default=True,
                        help='Use keypoint supervision')
    parser.add_argument('--no_kps', action='store_true', default=False,
                        help='Disable keypoint supervision')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log every N iterations')
    parser.add_argument('--save_interval', type=int, default=80,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    return parser.parse_args()


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """Linear warmup scheduler."""
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = x / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return LambdaLR(optimizer, f)


def step_lr_scheduler(optimizer, steps, gamma=0.1):
    """Step learning rate scheduler."""
    def f(x):
        decay = 1.0
        for step in steps:
            if x >= step:
                decay *= gamma
        return decay
    return LambdaLR(optimizer, f)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, args):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'args': vars(args),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, epoch, args, writer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_bbox = 0.0
    total_obj = 0.0
    total_kps = 0.0
    total_pos = 0

    for batch_idx, (images, gt_bboxes, gt_kpss) in enumerate(dataloader):
        images = images.to(device)

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss_dict = loss_fn(predictions, gt_bboxes, gt_kpss, images)
        loss = loss_dict['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Accumulate
        total_loss += loss.item()
        total_cls += loss_dict['loss_cls'].item()
        total_bbox += loss_dict['loss_bbox'].item()
        total_obj += loss_dict['loss_obj'].item()
        total_kps += loss_dict['loss_kps'].item()
        total_pos += loss_dict['num_pos']

        # Log
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_pos = total_pos / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                f"Cls: {loss_dict['loss_cls'].item():.4f} "
                f"Bbox: {loss_dict['loss_bbox'].item():.4f} "
                f"Obj: {loss_dict['loss_obj'].item():.4f} "
                f"Kps: {loss_dict['loss_kps'].item():.4f} "
                f"Pos: {loss_dict['num_pos']} "
                f"LR: {lr:.6f}"
            )

            # TensorBoard
            if writer:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/loss_cls', loss_dict['loss_cls'].item(), global_step)
                writer.add_scalar('train/loss_bbox', loss_dict['loss_bbox'].item(), global_step)
                writer.add_scalar('train/loss_obj', loss_dict['loss_obj'].item(), global_step)
                writer.add_scalar('train/loss_kps', loss_dict['loss_kps'].item(), global_step)
                writer.add_scalar('train/num_pos', loss_dict['num_pos'], global_step)
                writer.add_scalar('train/lr', lr, global_step)

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'loss_cls': total_cls / n,
        'loss_bbox': total_bbox / n,
        'loss_obj': total_obj / n,
        'loss_kps': total_kps / n,
        'avg_pos': total_pos / n,
    }


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    total_pos = 0

    for images, gt_bboxes, gt_kpss in dataloader:
        images = images.to(device)
        predictions = model(images)
        loss_dict = loss_fn(predictions, gt_bboxes, gt_kpss, images)
        total_loss += loss_dict['total'].item()
        total_pos += loss_dict['num_pos']

    n = max(len(dataloader), 1)
    return {'val_loss': total_loss / n, 'val_avg_pos': total_pos / n}


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data
    print(f"\nLoading training data from: {args.data_dir}")
    print(f"Annotations: {args.ann_file}")
    train_loader = create_dataloader(
        ann_file=args.ann_file,
        img_prefix=args.data_dir,
        batch_size=args.batch_size,
        input_size=tuple(args.input_size),
        augment=True,
        num_workers=args.num_workers,
        shuffle=True,
    )
    print(f"Training batches: {len(train_loader)}")

    val_loader = None
    if args.val_ann and args.val_dir:
        print(f"Loading validation data from: {args.val_dir}")
        val_loader = create_dataloader(
            ann_file=args.val_ann,
            img_prefix=args.val_dir,
            batch_size=args.batch_size,
            input_size=tuple(args.input_size),
            augment=False,
            num_workers=args.num_workers,
            shuffle=False,
        )

    # Model
    print("\nCreating YuNet model...")
    use_kps = args.use_kps and not args.no_kps
    model = YuNet(num_classes=1, use_kps=use_kps)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Loss
    loss_fn = YuNetLoss(num_classes=1, use_kps=use_kps)
    loss_fn = loss_fn.to(device)

    # Optimizer (SGD like the original config)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    # Original: warmup 1500 steps, step decays at 50*lr_mult=400 and 68*lr_mult=544 epochs
    # We scale down for practicality: warmup 5 epochs, step at 100, 200 epochs
    total_iters = len(train_loader) * args.epochs
    warmup_iters = len(train_loader) * args.warmup_epochs
    step1 = int(len(train_loader) * 0.6 * args.epochs)  # ~60% through
    step2 = int(len(train_loader) * 0.85 * args.epochs)  # ~85% through

    def lr_lambda(step):
        """Combined warmup + step schedule."""
        if step < warmup_iters:
            # Linear warmup
            return 0.001 + (1.0 - 0.001) * step / warmup_iters
        elif step < step1:
            return 1.0
        elif step < step2:
            return 0.1
        else:
            return 0.01

    scheduler = LambdaLR(optimizer, lr_lambda)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(logs_dir)) if SummaryWriter else None
    if writer:
        print(f"\nTensorBoard logs: {logs_dir}")

    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")

    # ============================================================
    #  Training Loop
    # ============================================================
    print(f"\nStarting training: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print("=" * 70)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            epoch, args, writer, device,
        )

        # Print epoch summary
        print(
            f"Epoch [{epoch}/{args.epochs}] Summary: "
            f"Loss={train_metrics['loss']:.4f} "
            f"Cls={train_metrics['loss_cls']:.4f} "
            f"Bbox={train_metrics['loss_bbox']:.4f} "
            f"Obj={train_metrics['loss_obj']:.4f} "
            f"Kps={train_metrics['loss_kps']:.4f} "
            f"AvgPos={train_metrics['avg_pos']:.1f}"
        )

        # Validation
        if val_loader is not None and epoch % 10 == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            print(f"  Validation: Loss={val_metrics['val_loss']:.4f}")
            if writer:
                writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics['val_loss'],
                    weights_dir / 'best_model.pth', args,
                )
                print(f"  [Best] model saved (val_loss={best_val_loss:.4f})")

        # Regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics['loss'],
                weights_dir / f'epoch_{epoch + 1}.pth', args,
            )

    # Save final model
    final_path = weights_dir / 'final_model.pth'
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, train_metrics['loss'],
        final_path, args,
    )
    print(f"\nFinal model saved: {final_path}")

    # Save ONNX model
    export_onnx(model, weights_dir / 'yunet_final.onnx', device, tuple(args.input_size))

    if writer:
        writer.close()
    print("Training complete!")


def export_onnx(model, output_path, device, input_size=(640, 640)):
    """Export model to ONNX format compatible with OpenCV FaceDetectorYN."""
    print(f"\nExporting ONNX model to: {output_path}")
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0], device=device)

    with torch.no_grad():
        # Test forward pass
        model(dummy_input)

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['cls_scores_0', 'cls_scores_1', 'cls_scores_2',
                         'bbox_preds_0', 'bbox_preds_1', 'bbox_preds_2',
                         'obj_preds_0', 'obj_preds_1', 'obj_preds_2',
                         'kps_preds_0', 'kps_preds_1', 'kps_preds_2'],
            dynamic_axes={'input': {2: 'height', 3: 'width'}},
        )
    print(f"ONNX model exported: {output_path}")


if __name__ == '__main__':
    main()

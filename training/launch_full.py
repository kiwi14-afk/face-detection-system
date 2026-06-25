"""Launch training on full WIDER FACE training set (preprocessed)."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
training_dir = Path(__file__).resolve().parent

data_dir = str(project_root / "training/data/WIDER_train_processed/images")
ann_file = str(training_dir / "data/splits_full/train_annotations.txt")
val_dir = data_dir  # Same image dir as training (split from training set)
val_ann = str(training_dir / "data/splits_full/val_annotations.txt")
output_dir = str(training_dir / "outputs/model_full")

sys.argv = [
    'train.py',
    '--data_dir', data_dir,
    '--ann_file', ann_file,
    '--val_dir', val_dir,
    '--val_ann', val_ann,
    '--output_dir', output_dir,
    '--batch_size', '16',
    '--epochs', '100',
    '--lr', '0.01',
    '--input_size', '640', '640',
    '--num_workers', '0',
    '--log_interval', '50',
    '--save_interval', '20',
    '--no_kps',
]

sys.path.insert(0, str(training_dir))
from yunet_torch.train import main

print(f"Training on: {data_dir}")
print(f"Train entries: 10,304 images")
print(f"Val: 2,576 images")
sys.exit(main())

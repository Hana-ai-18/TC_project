"""
Kaggle entry point for SRC-Track v2.

Usage:
  !python scripts/train_src.py \
      --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
      --output_dir   /kaggle/working/runs/src_v1 \
      --batch_size 32 --num_epochs 70 --learning_rate 1e-4 --use_amp
"""
import os, sys

# Ensure package is importable from Kaggle working dir
_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

from src_track.training.trainer import parse_args, _apply_aliases, train
from src_track.config import get_config

if __name__ == "__main__":
    args = parse_args()
    args = _apply_aliases(args)
    cfg  = get_config()

    # Auto-generate regime labels nếu chưa có
    import subprocess, sys as _sys
    os.makedirs(args.save_dir, exist_ok=True)
    regime_csv = os.path.join(args.save_dir, "sequence_regime_labels.csv")
    if not os.path.exists(regime_csv):
        print(f"  [INFO] regime_csv not found, generating...")
        gen_script = os.path.join(os.path.dirname(__file__), "generate_labels.py")
        if os.path.exists(gen_script):
            subprocess.run([_sys.executable, gen_script,
                "--data1d_dir", os.path.join(args.data_root, "Data1d"),
                "--data3d_dir",  os.path.join(args.data_root, "Data3d"),
                "--output", regime_csv], check=True)
        else:
            print(f"  [WARN] generate_labels.py not found, training without regime labels")

    # Apply CLI overrides
    data_root = args.data_root
    cfg.data.data1d_train   = os.path.join(data_root, "Data1d", "train")
    cfg.data.data1d_val     = os.path.join(data_root, "Data1d", "val")
    cfg.data.data1d_test    = os.path.join(data_root, "Data1d", "test")
    cfg.data.data3d_dir     = os.path.join(data_root, "Data3d")
    cfg.data.env_data_dir   = os.path.join(data_root, "Env_Data")
    cfg.train.save_dir      = args.save_dir
    cfg.train.max_epochs    = args.max_epochs
    cfg.train.batch_size    = args.batch_size
    cfg.train.lr            = args.lr
    cfg.train.num_workers   = args.num_workers
    cfg.train.seed          = args.seed
    if getattr(args, "no_adaptive_thresh", False):
        cfg.train.use_adaptive_thresh = False
    if getattr(args, "regime_csv", None):
        cfg.data.regime_label_csv = args.regime_csv
    else:
        # Default: put regime CSV in save_dir
        os.makedirs(args.save_dir, exist_ok=True)
        cfg.data.regime_label_csv = os.path.join(args.save_dir,
                                                   "sequence_regime_labels.csv")

    train(cfg, args)

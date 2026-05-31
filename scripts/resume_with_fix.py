"""
Resume từ checkpoint v3 với speed fix mạnh hơn.

Chạy từ Kaggle notebook đang chạy:
  !python scripts/resume_with_fix.py \
      --checkpoint /kaggle/working/runs/src_v3/best_ade.pth \
      --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
      --output_dir /kaggle/working/runs/src_v3b \
      --num_epochs 70 --use_amp
"""
import os, sys, torch
_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg not in sys.path: sys.path.insert(0, _pkg)

from src_track.training.trainer import parse_args, _apply_aliases, train
from src_track.config import get_config

if __name__ == "__main__":
    args = parse_args()
    args = _apply_aliases(args)
    cfg  = get_config()

    data_root = args.data_root
    cfg.data.data1d_train = os.path.join(data_root, "Data1d", "train")
    cfg.data.data1d_val   = os.path.join(data_root, "Data1d", "val")
    cfg.data.data3d_dir   = os.path.join(data_root, "Data3d")
    cfg.data.env_data_dir = os.path.join(data_root, "Env_Data")
    cfg.train.save_dir    = args.save_dir
    cfg.train.max_epochs  = args.max_epochs
    cfg.train.batch_size  = args.batch_size
    cfg.train.num_workers = args.num_workers
    cfg.train.seed        = args.seed
    cfg.train.use_amp     = args.use_amp

    # v3b speed fixes
    cfg.loss.w_speed   = 10.0   # was 5.0
    cfg.loss.w_regime  = 0.5
    cfg.loss.huber_delta = 300.
    cfg.train.lr       = 2e-4
    os.makedirs(args.save_dir, exist_ok=True)
    cfg.data.regime_label_csv = os.path.join(args.save_dir, "sequence_regime_labels.csv")
    # Copy regime csv from old run if exists
    old_csv = os.path.join(os.path.dirname(args.save_dir), "src_v3", "sequence_regime_labels.csv")
    if not os.path.exists(cfg.data.regime_label_csv) and os.path.exists(old_csv):
        import shutil
        shutil.copy(old_csv, cfg.data.regime_label_csv)
        print(f"  Copied regime CSV from {old_csv}")

    print("\n" + "="*60)
    print("  SPEED FIX v3b RESUME")
    print(f"  w_speed:   5.0 → 10.0")
    print(f"  SpeedHead: 48 → 93 km/6h init (gt_mean=113)")
    print(f"  checkpoint: {args.resume}")
    print("="*60 + "\n")

    train(cfg, args)

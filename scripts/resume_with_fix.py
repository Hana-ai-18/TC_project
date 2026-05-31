"""
Resume từ checkpoint hiện tại với speed fix triệt để.

FIXES:
  FIX1: w_speed=5.0 × learned_weight (was: learned_weight only, 5.0 bị ignore!)
  FIX2: huber_delta 100→300
  FIX3: w_regime 0.5→1.0 (RC cần nhiều gradient hơn)
  FIX4: ConstrainedLossWeights.w_speed cap 2.0→5.0

Chạy:
  !python scripts/resume_with_fix.py \
      --checkpoint /kaggle/working/runs/src_v1/best_ade.pth \
      --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
      --output_dir /kaggle/working/runs/src_v4_speedfix \
      --num_epochs 70 --use_amp
"""
import os, sys, shutil
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
    cfg.train.use_amp     = getattr(args, 'use_amp', False)

    # SPEED FIX
    cfg.loss.w_speed    = 5.0    # FIX: now ACTUALLY used (bug fixed)
    cfg.loss.w_regime   = 1.0    # was 0.5 → RC cần gradient mạnh hơn  
    cfg.loss.huber_delta = 300.  # was 100 → match ADE scale
    cfg.train.lr        = 2e-4

    os.makedirs(args.save_dir, exist_ok=True)
    # Copy regime csv từ run cũ
    for old_dir in ["src_v1", "src_v3", "src_v2"]:
        old_csv = f"/kaggle/working/runs/{old_dir}/sequence_regime_labels.csv"
        new_csv = os.path.join(args.save_dir, "sequence_regime_labels.csv")
        if os.path.exists(old_csv) and not os.path.exists(new_csv):
            shutil.copy(old_csv, new_csv)
            print(f"  Copied regime CSV from {old_csv}")
            break
    cfg.data.regime_label_csv = os.path.join(args.save_dir, "sequence_regime_labels.csv")

    print("\n" + "="*65)
    print("  SPEED FIX v4 — Fixes 4 bugs causing ADE plateau at 300km")
    print(f"  FIX1: w_speed = learned × 5.0 (was: 5.0 ignored in forward!)")
    print(f"  FIX2: huber_delta = 300 (was 100, wrong scale)")
    print(f"  FIX3: w_regime = 1.0 (was 0.5, RC needs more gradient)")
    print(f"  FIX4: w_speed_cap = 5.0 (was 2.0)")
    print(f"  checkpoint: {args.resume}")
    print("="*65 + "\n")
    train(cfg, args)

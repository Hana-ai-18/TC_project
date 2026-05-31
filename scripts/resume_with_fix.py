"""
Resume training từ best checkpoint với speed fix.

QUAN TRỌNG: Sau khi resume, SpeedHead bias được reset về 9.0
(target 48 km/6h thay vì 18 km/6h).

Chạy:
  !python scripts/resume_with_fix.py \
      --checkpoint /kaggle/working/runs/src_v1/best_ade.pth \
      --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
      --output_dir /kaggle/working/runs/src_v2_speedfix \
      --num_epochs 70 --use_amp
"""
import os, sys, torch

_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg not in sys.path: sys.path.insert(0, _pkg)

from src_track.training.trainer import parse_args, _apply_aliases, train
from src_track.config import get_config

def reset_speed_head(model, target_bias=9.0):
    """Reset SpeedHead output bias → target 48 km/6h."""
    with torch.no_grad():
        model.speed_head.net[-1].bias.fill_(target_bias)
    pred = float(torch.nn.functional.softplus(
        torch.tensor(target_bias)) * 5.0 + 3.0)
    print(f"  SpeedHead bias reset: {target_bias} → output ≈ {pred:.1f} km/6h")

if __name__ == "__main__":
    args = parse_args()
    args = _apply_aliases(args)
    cfg  = get_config()

    data_root = args.data_root
    cfg.data.data1d_train   = os.path.join(data_root, "Data1d", "train")
    cfg.data.data1d_val     = os.path.join(data_root, "Data1d", "val")
    cfg.data.data1d_test    = os.path.join(data_root, "Data1d", "test")
    cfg.data.data3d_dir     = os.path.join(data_root, "Data3d")
    cfg.data.env_data_dir   = os.path.join(data_root, "Env_Data")
    cfg.train.save_dir      = args.save_dir
    cfg.train.max_epochs    = args.max_epochs
    cfg.train.batch_size    = args.batch_size
    cfg.train.num_workers   = args.num_workers
    cfg.train.seed          = args.seed

    # ── SPEED FIX PARAMS ──────────────────────────────────────
    cfg.loss.huber_delta  = 300.0   # was 50 → match ADE scale
    cfg.loss.w_speed      = 5.0     # was 0.5 → 10x to compete
    cfg.loss.w_regime     = 0.5     # was 0.1 → RC needs more signal
    cfg.train.lr          = 2e-4    # was 1e-4 → faster escape
    cfg.train.use_amp     = args.use_amp
    if getattr(args, "regime_csv", None):
        cfg.data.regime_label_csv = args.regime_csv
    else:
        cfg.data.regime_label_csv = os.path.join(
            args.save_dir, "sequence_regime_labels.csv")

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*65)
    print("  SPEED FIX RESUME")
    print(f"  checkpoint: {args.resume}")
    print(f"  huber_delta: 50 → 300")
    print(f"  w_speed:     0.5 → 5.0  (10x amplify speed loss)")
    print(f"  w_regime:    0.1 → 0.5  (RC gradient amplify)")
    print(f"  lr:          1e-4 → 2e-4")
    print(f"  SpeedHead init: 18 → 48 km/6h (SCS true mean)")
    print("="*65 + "\n")

    # Run training (resume handled inside train())
    train(cfg, args)

"""
scripts/train_src.py
Notebook-style entry point for SRC-Track v2 training.
Accepts Kaggle-style args (--dataset_root, --output_dir, --learning_rate, --num_epochs)
AND new-style args (--data_root, --save_dir, --lr, --max_epochs).

Usage (Kaggle):
  python scripts/train_src.py \
      --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
      --output_dir   /kaggle/working/runs/src_v1 \
      --batch_size   32 \
      --num_epochs   70 \
      --learning_rate 1e-4 \
      --use_amp \
      --regime_csv  /kaggle/working/runs/src_v1/sequence_regime_labels.csv
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="SRC-Track v2 Training")

    # ── Data paths ────────────────────────────────────────────
    p.add_argument("--dataset_root",  default=None,
                   help="Root dir containing Data1d/, Data3d/, Env_Data/")
    p.add_argument("--data_root",     default=None,
                   help="Alias for --dataset_root")

    # ── Output ────────────────────────────────────────────────
    p.add_argument("--output_dir",  default=None,
                   help="Where to save checkpoints")
    p.add_argument("--save_dir",    default="checkpoints/v2/",
                   help="Alias for --output_dir")

    # ── Regime labels ─────────────────────────────────────────
    p.add_argument("--regime_csv",  default=None,
                   help="Path to sequence_regime_labels.csv. "
                        "If None, auto-detect in data_root/")

    # ── Training hyperparams ──────────────────────────────────
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--num_epochs",    type=int,   default=70,
                   help="Number of training epochs")
    p.add_argument("--max_epochs",    type=int,   default=None,
                   help="Alias for --num_epochs")
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="Initial learning rate")
    p.add_argument("--lr",            type=float, default=None,
                   help="Alias for --learning_rate")
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--num_workers",   type=int,   default=2,
                   help="DataLoader workers (Kaggle: 2 is safe)")
    p.add_argument("--use_amp",       action="store_true",
                   help="Enable AMP mixed precision. May cause NaN on some GPUs - disable if loss=nan")
    p.add_argument("--no_amp",        action="store_true",
                   help="Force disable AMP (use if loss=nan with --use_amp)")
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--resume",        default=None,
                   help="Checkpoint path to resume training")
    p.add_argument("--no_adaptive_thresh", action="store_true",
                   help="Disable adaptive difficulty threshold")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve aliases ────────────────────────────────────────
    data_root = args.data_root or args.dataset_root
    if not data_root:
        print("ERROR: provide --data_root or --dataset_root")
        sys.exit(1)

    save_dir  = args.output_dir or args.save_dir
    lr        = args.lr or args.learning_rate
    max_epochs = args.max_epochs or args.num_epochs

    # Auto-detect regime CSV
    regime_csv = args.regime_csv
    if regime_csv is None:
        auto_candidates = [
            os.path.join(save_dir, "sequence_regime_labels.csv"),
            os.path.join(data_root, "sequence_regime_labels.csv"),
            os.path.join(os.path.dirname(save_dir), "sequence_regime_labels.csv"),
            "/kaggle/working/sequence_regime_labels.csv",
        ]
        for cand in auto_candidates:
            if os.path.exists(cand):
                regime_csv = cand
                print(f"[auto] Found regime_csv: {regime_csv}")
                break
        if regime_csv is None:
            regime_csv = os.path.join(save_dir, "sequence_regime_labels.csv")
            print(f"[warn] regime_csv not found. Will use: {regime_csv}")
            print(f"       Run generate_labels.py first if labels are missing.")

    print(f"\n{'='*65}")
    print(f"  SRC-Track v2 — Kaggle Training")
    print(f"  data_root  : {data_root}")
    print(f"  save_dir   : {save_dir}")
    print(f"  regime_csv : {regime_csv}")
    print(f"  lr={lr}  bs={args.batch_size}  epochs={max_epochs}  amp={args.use_amp}")
    print(f"{'='*65}\n")

    # ── Build config ──────────────────────────────────────────
    from src_track.config import get_config
    cfg = get_config()

    cfg.data.data1d_train     = os.path.join(data_root, "Data1d", "train")
    cfg.data.data1d_val       = os.path.join(data_root, "Data1d", "val")
    cfg.data.data1d_test      = os.path.join(data_root, "Data1d", "test")
    cfg.data.data3d_dir       = os.path.join(data_root, "Data3d")
    cfg.data.env_data_dir     = os.path.join(data_root, "Env_Data")
    cfg.data.regime_label_csv = regime_csv

    cfg.train.save_dir    = save_dir
    cfg.train.lr          = lr
    cfg.train.batch_size  = args.batch_size
    cfg.train.max_epochs  = max_epochs
    cfg.train.num_workers = args.num_workers
    cfg.train.grad_clip   = args.grad_clip
    cfg.train.patience    = args.patience
    cfg.train.seed        = args.seed
    cfg.train.use_adaptive_thresh = not args.no_adaptive_thresh

    # Verify paths exist
    missing = []
    for attr, path in [
        ("Data1d/train", cfg.data.data1d_train),
        ("Data3d",       cfg.data.data3d_dir),
        ("Env_Data",     cfg.data.env_data_dir),
    ]:
        if not os.path.exists(path):
            missing.append(f"  MISSING: {attr} → {path}")
    if missing:
        print("[ERROR] Some data paths not found:")
        for m in missing: print(m)
        sys.exit(1)
    else:
        print("[OK] All data paths found")

    os.makedirs(save_dir, exist_ok=True)

    # ── Run training ──────────────────────────────────────────
    from src_track.training.trainer import train, set_seed
    set_seed(cfg.train.seed)

    # Pass use_amp flag (not in config, passed via args-like object)
    class _args:
        use_amp = args.use_amp
        resume  = args.resume

    best_ade = train(cfg, _args())
    print(f"\n[Done] Best ADE: {best_ade:.1f} km")


if __name__ == "__main__":
    main()

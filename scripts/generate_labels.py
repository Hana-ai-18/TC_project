"""
scripts/generate_labels.py
Thin wrapper — notebook-style entry point for generate_regime_labels.py.
Accepts BOTH old-style args (--data1d_dir, --data3d_dir) AND new-style (--data_root).

Usage (Kaggle):
  python scripts/generate_labels.py \
      --data1d_dir /kaggle/.../Data1d/train \
      --data3d_dir /kaggle/.../Data3d \
      --output     /kaggle/working/runs/sequence_regime_labels.csv

Usage (new style):
  python scripts/generate_labels.py \
      --data_root /kaggle/.../tc-ofm \
      --output    /kaggle/working/runs/sequence_regime_labels.csv
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    # New-style: single root
    p.add_argument("--data_root",   default=None,
                   help="Root containing Data1d/, Data3d/, Env_Data/")
    # Old-style: individual dirs
    p.add_argument("--data1d_dir",  default=None,
                   help="Path to Data1d/train (or Data1d/)")
    p.add_argument("--data3d_dir",  default=None,
                   help="Path to Data3d/")
    p.add_argument("--env_data_dir",default=None,
                   help="Path to Env_Data/")
    p.add_argument("--output",   default="sequence_regime_labels.csv")
    p.add_argument("--obs_len",  type=int,   default=8)
    p.add_argument("--pred_len", type=int,   default=12)
    p.add_argument("--stride",   type=int,   default=1)
    p.add_argument("--smooth_window", type=int, default=3)
    p.add_argument("--inner_deg", type=float, default=3.0)
    p.add_argument("--outer_deg", type=float, default=7.0)
    p.add_argument("--splits",    nargs="+",
                   default=["train", "val", "test"])
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve data root
    if args.data_root:
        data_root = args.data_root
        data1d_root = os.path.join(data_root, "Data1d")
        data3d_dir  = os.path.join(data_root, "Data3d")
    elif args.data1d_dir:
        # data1d_dir might be Data1d/train — go up to Data1d/
        d = args.data1d_dir
        # if it ends with a split name, use parent as data1d root
        if os.path.basename(d) in ("train", "val", "test"):
            data1d_root = os.path.dirname(d)
        else:
            data1d_root = d
        data3d_dir = args.data3d_dir or os.path.join(
            os.path.dirname(data1d_root), "Data3d")
        data_root  = os.path.dirname(data1d_root)
    else:
        print("ERROR: provide --data_root OR --data1d_dir")
        sys.exit(1)

    print(f"\n[generate_labels]")
    print(f"  data1d_root : {data1d_root}")
    print(f"  data3d_dir  : {data3d_dir}")
    print(f"  output      : {args.output}")
    print(f"  splits      : {args.splits}")

    import pandas as pd
    from src_track.data.dataset import (
        load_data1d_directory,
        compute_annular_steering,
        classify_regime,
        compute_rii,
        decode_data1d,
    )
    import numpy as np

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    all_records = []
    window      = args.obs_len + args.pred_len

    for split in args.splits:
        split_dir = os.path.join(data1d_root, split)
        if not os.path.exists(split_dir):
            print(f"  [skip] {split_dir} not found")
            continue

        # Load all storms in this split
        storms = {}
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".txt"):
                continue
            sid  = os.path.splitext(fname)[0]   # e.g. "1975_RITA"
            rows = []
            with open(os.path.join(split_dir, fname),
                      encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or parts[0].startswith('#'):
                        continue
                    try:
                        int(parts[0])   # frame_id
                        vals = list(map(float, parts[1:5]))
                    except ValueError:
                        try:
                            vals = list(map(float, parts[1:5]))
                        except ValueError:
                            continue
                    rows.append(vals)
            if rows:
                storms[sid] = np.array(rows, dtype=np.float32)

        print(f"  {split}: {len(storms)} storms")

        for sid, raw in storms.items():
            T = len(raw)
            if T < window:
                continue

            # Parse year and name from sid (e.g. "1970_0003")
            year = sid.split("_")[0] if "_" in sid else sid[:4]
            name = sid.split("_")[1] if "_" in sid else sid[4:]

            # Load ERA5 for this storm
            d3d_full = None
            for cand in [
                os.path.join(data3d_dir, year, name),
                os.path.join(data3d_dir, year, sid),
                os.path.join(data3d_dir, sid),
            ]:
                if os.path.isdir(cand):
                    files = sorted(
                        f for f in os.listdir(cand) if f.endswith(".npy"))
                    if files:
                        patches = []
                        for fp in files:
                            arr = np.load(os.path.join(cand, fp)).astype(np.float32)
                            if arr.ndim == 3 and arr.shape[-1] == 13:
                                arr = arr.transpose(2, 0, 1)   # [H,W,C]→[C,H,W]
                            patches.append(arr[:13])
                        d3d_full = patches
                    break

            for start in range(0, T - window + 1, args.stride):
                obs = raw[start: start + args.obs_len]

                # Compute steering angles per obs step
                angles = []
                if d3d_full is not None and start + args.obs_len <= len(d3d_full):
                    for t in range(start, start + args.obs_len):
                        patch = d3d_full[t]
                        # U500=ch5, V500=ch9 (TCND order)
                        if patch.shape[0] >= 10:
                            u500 = patch[5]; v500 = patch[9]
                            _, _, ang, _ = compute_annular_steering(
                                u500, v500,
                                inner_deg=args.inner_deg,
                                outer_deg=args.outer_deg,
                            )
                            angles.append(ang)

                if not angles:
                    # Fallback: kinematic direction from best-track
                    phys = decode_data1d(obs)
                    if args.obs_len >= 2:
                        import math
                        dlat = phys[-1, 1] - phys[-2, 1]
                        dlon = (phys[-1, 0] - phys[-2, 0]) * np.cos(np.deg2rad(phys[-1, 1]))
                        ang  = math.degrees(math.atan2(dlon, dlat))
                        angles = [ang] * args.obs_len

                angles_s   = pd.Series(angles).rolling(
                    args.smooth_window, min_periods=1).mean().values
                last_angle = float(angles_s[-1])
                regime     = classify_regime(last_angle)
                rii        = compute_rii(list(angles_s),
                                         window=args.smooth_window)

                phys      = decode_data1d(obs)
                speeds    = np.sqrt(
                    np.diff(phys[:, 0])**2 + np.diff(phys[:, 1])**2) * 111.0
                speed_var = float(speeds.std()) if len(speeds) > 0 else 0.0

                rii_s     = min(rii / 2.0, 1.0)
                reg_s     = {0: 0.2, 1: 1.0, 2: 0.4}.get(regime, 0.5)
                spd_s     = min(speed_var / 15.0, 1.0)
                difficulty = 0.5 * rii_s + 0.3 * reg_s + 0.2 * spd_s

                all_records.append({
                    "storm_id":          sid,
                    "start_idx":         start,
                    "regime":            regime,
                    "rii":               rii,
                    "steering_angle_deg": last_angle,
                    "difficulty":        difficulty,
                })

    df = pd.DataFrame(all_records)
    df.to_csv(args.output, index=False)
    print(f"\n[Done] {len(df)} sequences → {args.output}")

    if len(df) > 0:
        print(f"\nRegime distribution:")
        for r, rn in [(0,"A"),(1,"B"),(2,"C")]:
            n = (df["regime"]==r).sum()
            print(f"  Regime {rn}: {n} ({100*n/len(df):.1f}%)")
        print(f"\nDifficulty distribution:")
        print(f"  Easy  (diff<0.4): {(df['difficulty']<0.4).sum()} ({100*(df['difficulty']<0.4).mean():.1f}%)")
        print(f"  Hard  (diff>0.7): {(df['difficulty']>0.7).sum()} ({100*(df['difficulty']>0.7).mean():.1f}%)")
        print(f"  RII>1.0 (unstable): {(df['rii']>1.0).sum()} ({100*(df['rii']>1.0).mean():.1f}%)")


if __name__ == "__main__":
    main()

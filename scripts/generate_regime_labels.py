"""
scripts/generate_regime_labels.py — Run ONCE before training

Generates sequence_regime_labels.csv with:
  storm_id, start_idx, regime (0=A/1=B/2=C), rii, steering_angle_deg

Requires: Data1d/ and Data3d/ under data_root.

Usage:
  python scripts/generate_regime_labels.py \\
      --data_root TCND_vn/ \\
      --output    sequence_regime_labels.csv
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src_track.data.dataset import (
    compute_annular_steering,
    classify_regime,
    compute_rii,
    decode_data1d,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="TCND_vn/")
    p.add_argument("--output",    default="sequence_regime_labels.csv")
    p.add_argument("--obs_len",   type=int, default=8)
    p.add_argument("--pred_len",  type=int, default=12)
    p.add_argument("--stride",    type=int, default=1)
    p.add_argument("--smooth_window", type=int, default=3)
    p.add_argument("--inner_deg", type=float, default=3.0)
    p.add_argument("--outer_deg", type=float, default=7.0)
    p.add_argument("--splits",    nargs="+", default=["train", "val", "test"])
    return p.parse_args()


def load_data1d_files(data1d_split_dir: str):
    """Load all .txt files in a split dir. Returns {storm_id: [T, 4] array}."""
    storms = {}
    for fname in sorted(os.listdir(data1d_split_dir)):
        if not fname.endswith(".txt"):
            continue
        sid = os.path.splitext(fname)[0]
        rows = []
        with open(os.path.join(data1d_split_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    # Format: frame_id lon_n lat_n pres_n wnd_n date name
                    # Or: lon_n lat_n pres_n wnd_n
                    if len(parts) >= 5:
                        vals = [float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4])]
                    elif len(parts) == 4:
                        vals = list(map(float, parts))
                    else:
                        continue
                    rows.append(vals)
                except ValueError:
                    continue
        if rows:
            storms[sid] = np.array(rows, dtype=np.float32)
    return storms


def compute_labels_for_storm(
    sid: str,
    raw: np.ndarray,       # [T, 4] normalized
    data3d_dir: str,
    obs_len: int,
    pred_len: int,
    stride: int,
    smooth_window: int,
    inner_deg: float,
    outer_deg: float,
) -> list:
    window = obs_len + pred_len
    T = len(raw)
    records = []

    # Try loading ERA5
    d3d_full = None
    for candidate in [
        os.path.join(data3d_dir, f"{sid}.npy"),
        os.path.join(data3d_dir, sid, f"{sid}.npy"),
    ]:
        if os.path.exists(candidate):
            try:
                d3d_full = np.load(candidate, mmap_mode='r')
            except Exception:
                pass
            break

    for start in range(0, T - window + 1, stride):
        obs = raw[start: start + obs_len]

        # Compute steering angles per obs step
        angles = []
        if d3d_full is not None and start + obs_len <= len(d3d_full):
            d3d_obs = d3d_full[start: start + obs_len]
            for t in range(obs_len):
                patch = d3d_obs[t]   # [C, H, W] or [H, W, C]
                # Handle channel-last vs channel-first
                if patch.ndim == 3:
                    if patch.shape[0] == 13:   # channel-first
                        u500 = patch[5]         # U500
                        v500 = patch[9]         # V500
                    else:                       # channel-last
                        u500 = patch[:, :, 5]
                        v500 = patch[:, :, 9]
                    _, _, ang, _ = compute_annular_steering(
                        u500, v500,
                        inner_deg=inner_deg, outer_deg=outer_deg
                    )
                    angles.append(ang)

        if not angles:
            # Fallback: use kinematic direction from best-track
            phys = decode_data1d(obs)
            if obs_len >= 2:
                dlat = phys[-1, 1] - phys[-2, 1]
                dlon = (phys[-1, 0] - phys[-2, 0]) * np.cos(np.deg2rad(phys[-1, 1]))
                import math
                ang = math.degrees(math.atan2(dlon, dlat))
                angles = [ang] * obs_len

        # Temporal smoothing
        if len(angles) >= smooth_window:
            angles_smooth = pd.Series(angles).rolling(
                smooth_window, min_periods=1).mean().values
        else:
            angles_smooth = np.array(angles)

        last_angle = float(angles_smooth[-1]) if len(angles_smooth) > 0 else 0.0
        regime     = classify_regime(last_angle)
        rii        = compute_rii(list(angles_smooth), window=smooth_window)

        # Speed variance for difficulty
        phys      = decode_data1d(obs)
        speeds    = np.sqrt(np.diff(phys[:, 0])**2 + np.diff(phys[:, 1])**2) * 111.0
        speed_var = float(speeds.std()) if len(speeds) > 0 else 0.0

        # Composite difficulty score
        rii_score    = min(rii / 2.0, 1.0)
        regime_score = {0: 0.2, 1: 1.0, 2: 0.4}.get(regime, 0.5)
        speed_score  = min(speed_var / 15.0, 1.0)
        difficulty   = 0.5 * rii_score + 0.3 * regime_score + 0.2 * speed_score

        records.append({
            "storm_id":          sid,
            "start_idx":         start,
            "regime":            regime,
            "rii":               rii,
            "steering_angle_deg": last_angle,
            "difficulty":        difficulty,
        })
    return records


def main():
    args = parse_args()
    data1d_root = os.path.join(args.data_root, "Data1d")
    data3d_dir  = os.path.join(args.data_root, "Data3d")

    all_records = []
    for split in args.splits:
        split_dir = os.path.join(data1d_root, split)
        if not os.path.exists(split_dir):
            print(f"  [skip] {split_dir} not found")
            continue
        storms = load_data1d_files(split_dir)
        print(f"  {split}: {len(storms)} storms")
        for sid, raw in storms.items():
            recs = compute_labels_for_storm(
                sid, raw, data3d_dir,
                args.obs_len, args.pred_len, args.stride,
                args.smooth_window, args.inner_deg, args.outer_deg,
            )
            all_records.extend(recs)
        print(f"  {split}: {sum(1 for r in all_records if True)} records so far")

    df = pd.DataFrame(all_records)
    df.to_csv(args.output, index=False)
    print(f"\n[Done] {len(df)} sequences → {args.output}")

    # Distribution stats
    if len(df) > 0:
        print(f"\nRegime distribution:")
        for r, rn in [(0,"A"),(1,"B"),(2,"C")]:
            n = (df["regime"]==r).sum()
            print(f"  Regime {rn}: {n} ({100*n/len(df):.1f}%)")
        print(f"\nRII distribution:")
        print(f"  Easy (RII<0.5): {(df['rii']<0.5).sum()} ({100*(df['rii']<0.5).mean():.1f}%)")
        print(f"  Hard (RII>1.0): {(df['rii']>1.0).sum()} ({100*(df['rii']>1.0).mean():.1f}%)")


if __name__ == "__main__":
    main()

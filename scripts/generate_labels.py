"""
scripts/generate_labels.py — SELF-CONTAINED, no external imports
Generates sequence_regime_labels.csv for SRC-Track training.

Works from any working directory (TC_project/, src_track_new/, etc.)

Usage:
  python scripts/generate_labels.py \
      --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
      --output       /kaggle/working/runs/src_v1/sequence_regime_labels.csv
"""
from __future__ import annotations
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd


# ── Inline physics helpers (no src_track import needed) ───────────

def decode_data1d(raw: np.ndarray) -> np.ndarray:
    out = np.empty_like(raw)
    out[..., 0] = (raw[..., 0] * 50 + 1800) / 10   # lon °E
    out[..., 1] = (raw[..., 1] * 50) / 10           # lat °N
    out[..., 2] = raw[..., 2] * 50 + 960
    out[..., 3] = raw[..., 3] * 25 + 40
    return out


def compute_annular_steering(u_patch, v_patch, center=(40, 40),
                              inner_deg=3.0, outer_deg=7.0, cell_deg=0.25):
    inner = inner_deg / cell_deg
    outer = outer_deg / cell_deg
    H, W  = u_patch.shape
    cy, cx = center
    y = np.arange(H).reshape(-1, 1)
    x = np.arange(W).reshape(1, -1)
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    mask = (dist >= inner) & (dist <= outer)
    n = mask.sum()
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    u_s = float((u_patch * mask).sum() / n)
    v_s = float((v_patch * mask).sum() / n)
    return u_s, v_s, float(np.degrees(np.arctan2(v_s, u_s))), float(np.sqrt(u_s**2 + v_s**2))


def classify_regime(angle_deg: float) -> int:
    a = angle_deg
    if -180.0 <= a <= -45.0:  return 0   # A: subtropical high
    elif -45.0 < a <= 30.0:   return 1   # B: transition
    elif 30.0  < a <= 135.0:  return 2   # C: westerlies
    return 1


def compute_rii(angle_history, window=3):
    if len(angle_history) < window:
        return 0.0
    angles = np.deg2rad(angle_history[-window:])
    r = np.clip(np.sqrt(np.sin(angles).mean()**2 + np.cos(angles).mean()**2), 1e-8, 1.0)
    return float(np.sqrt(-2.0 * np.log(r)) / (math.pi / 4))


# ── Arg parser ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate sequence_regime_labels.csv for SRC-Track")
    # Accept both naming styles
    p.add_argument("--dataset_root", default=None,
                   help="Root dir with Data1d/, Data3d/, Env_Data/  (Kaggle style)")
    p.add_argument("--data_root",    default=None,
                   help="Alias for --dataset_root")
    # Fine-grained overrides
    p.add_argument("--data1d_dir",   default=None,
                   help="Path to Data1d/ (overrides dataset_root)")
    p.add_argument("--data3d_dir",   default=None,
                   help="Path to Data3d/ (overrides dataset_root)")
    p.add_argument("--output",  default="sequence_regime_labels.csv")
    p.add_argument("--obs_len", type=int,   default=8)
    p.add_argument("--pred_len",type=int,   default=12)
    p.add_argument("--stride",  type=int,   default=1)
    p.add_argument("--smooth_window", type=int, default=3)
    p.add_argument("--inner_deg", type=float, default=3.0)
    p.add_argument("--outer_deg", type=float, default=7.0)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve data root (accept either arg name)
    root = args.dataset_root or args.data_root
    if not root and not args.data1d_dir:
        print("ERROR: provide --dataset_root or --data1d_dir")
        sys.exit(1)

    data1d_root = args.data1d_dir or os.path.join(root, "Data1d")
    data3d_dir  = args.data3d_dir or os.path.join(root, "Data3d")

    print(f"\n[generate_labels]")
    print(f"  data1d_root : {data1d_root}")
    print(f"  data3d_dir  : {data3d_dir}")
    print(f"  output      : {args.output}")
    print(f"  splits      : {args.splits}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    all_records = []
    window = args.obs_len + args.pred_len

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
            sid  = os.path.splitext(fname)[0]   # e.g. "1970_0003"
            rows = []
            with open(os.path.join(split_dir, fname),
                      encoding="utf-8", errors="ignore") as f:
                for line in f:
                    p = line.strip().split()
                    if not p or p[0].startswith("#"):
                        continue
                    try:
                        int(p[0])                          # frame_id
                        vals = list(map(float, p[1:5]))    # lon_n lat_n pres_n wnd_n
                        ts   = p[5] if len(p) > 5 else ""  # timestamp
                    except (ValueError, IndexError):
                        continue
                    rows.append((vals, ts))
            if rows:
                raw_arr  = np.array([r[0] for r in rows], dtype=np.float32)
                ts_list  = [r[1] for r in rows]
                storms[sid] = {"raw": raw_arr, "timestamps": ts_list}

        print(f"  {split}: {len(storms)} storms", end="", flush=True)

        for sid, sdata in storms.items():
            raw = sdata["raw"]
            ts  = sdata["timestamps"]
            T   = len(raw)
            if T < window:
                continue

            year = sid.split("_")[0] if "_" in sid else sid[:4]
            name = sid.split("_")[1] if "_" in sid else sid[4:]

            # Load ERA5 patches for this storm
            d3d_patches = _load_storm_d3d(data3d_dir, year, name, ts)

            for start in range(0, T - window + 1, args.stride):
                obs    = raw[start: start + args.obs_len]
                obs_ts = ts[start: start + args.obs_len]

                # Compute steering angles per obs step
                angles = []
                for i, t_stamp in enumerate(obs_ts):
                    patch = d3d_patches.get(t_stamp)
                    if patch is not None and patch.shape[0] >= 10:
                        # TCND channels: U500=ch5, V500=ch9
                        _, _, ang, _ = compute_annular_steering(
                            patch[5], patch[9],
                            inner_deg=args.inner_deg,
                            outer_deg=args.outer_deg)
                        angles.append(ang)

                if not angles:
                    # Fallback: kinematic direction from best-track
                    phys = decode_data1d(obs)
                    if args.obs_len >= 2:
                        dlat = phys[-1, 1] - phys[-2, 1]
                        dlon = (phys[-1, 0] - phys[-2, 0]) * np.cos(np.deg2rad(phys[-1, 1]))
                        ang  = math.degrees(math.atan2(dlon, dlat))
                        angles = [ang] * args.obs_len

                # Temporal smoothing
                angles_s = (pd.Series(angles)
                              .rolling(args.smooth_window, min_periods=1)
                              .mean().values)

                last_angle = float(angles_s[-1])
                regime     = classify_regime(last_angle)
                rii        = compute_rii(list(angles_s), window=args.smooth_window)

                phys = decode_data1d(obs)
                spds = np.sqrt(np.diff(phys[:, 0])**2 + np.diff(phys[:, 1])**2) * 111.0
                sv   = float(spds.std()) if len(spds) > 0 else 0.0

                difficulty = (0.5 * min(rii / 2.0, 1.0)
                              + 0.3 * {0: 0.2, 1: 1.0, 2: 0.4}.get(regime, 0.5)
                              + 0.2 * min(sv / 15.0, 1.0))

                all_records.append({
                    "storm_id":          sid,
                    "start_idx":         start,
                    "regime":            regime,
                    "rii":               round(rii, 4),
                    "steering_angle_deg": round(last_angle, 2),
                    "difficulty":        round(difficulty, 4),
                })

        print(f" → {sum(1 for r in all_records if True)} seqs so far")

    df = pd.DataFrame(all_records)
    df.to_csv(args.output, index=False)
    print(f"\n[Done] {len(df)} sequences → {args.output}")

    if len(df) > 0:
        print(f"\nRegime distribution:")
        for r, rn in [(0, "A"), (1, "B"), (2, "C")]:
            n = (df["regime"] == r).sum()
            print(f"  Regime {rn}: {n:5d} ({100*n/len(df):.1f}%)")
        print(f"\nDifficulty:")
        print(f"  Easy  (diff<0.4): {(df['difficulty']<0.4).sum()}")
        print(f"  Hard  (diff>0.7): {(df['difficulty']>0.7).sum()}")


def _load_storm_d3d(data3d_dir, year, name, timestamps):
    """
    Load Data3d patches for a storm.
    Returns dict {timestamp_str: patch_chw} where patch_chw is [13,81,81].
    """
    result = {}
    # Find storm directory
    storm_dir = None
    for try_name in [name, name.lstrip("0") or "0", name.zfill(4), name.zfill(2)]:
        d = os.path.join(data3d_dir, year, try_name)
        if os.path.isdir(d):
            storm_dir = d
            break

    if storm_dir is None:
        return result

    # Build filename → path map
    fname_map = {}
    try:
        for fn in os.listdir(storm_dir):
            if fn.endswith(".npy"):
                fname_map[fn] = os.path.join(storm_dir, fn)
    except OSError:
        return result

    for ts in timestamps:
        if not ts:
            continue
        # Try exact filename patterns
        patch = None
        for fn_try in [
            f"WP{year}{name}_{ts}.npy",
            f"WP{year}{name.zfill(4)}_{ts}.npy",
            f"{ts}.npy",
        ]:
            if fn_try in fname_map:
                patch = _read_npy_patch(fname_map[fn_try])
                break

        # Fuzzy match: any file containing the timestamp
        if patch is None:
            for fn, fp in fname_map.items():
                if ts in fn:
                    patch = _read_npy_patch(fp)
                    break

        if patch is not None:
            result[ts] = patch

    return result


def _read_npy_patch(fpath):
    """Load one .npy → [13, 81, 81] float32."""
    try:
        arr = np.load(fpath).astype(np.float32)
        if arr.ndim == 3:
            if arr.shape[-1] == 13:          # [H,W,C] → [C,H,W]
                arr = arr.transpose(2, 0, 1)
            if arr.shape[1] != 81 or arr.shape[2] != 81:
                # Simple crop/pad to 81×81
                h, w = arr.shape[1], arr.shape[2]
                if h < 81: arr = np.pad(arr, ((0,0),(0,81-h),(0,0)))
                if w < 81: arr = np.pad(arr, ((0,0),(0,0),(0,81-w)))
                arr = arr[:, :81, :81]
            return arr[:13]
    except Exception:
        pass
    return None


if __name__ == "__main__":
    main()

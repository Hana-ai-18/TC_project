"""
scripts/run_inference.py — SRC-Track v2

Full test-set evaluation + Sensitivity Ellipse Calibration (SEC).

Usage:
  python scripts/run_inference.py \\
      --data_root TCND_vn/ \\
      --checkpoint checkpoints/v2/best_ade.pth \\
      --output_dir results/v2/
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src_track.config import get_config
from src_track.data.dataset import SRCTrackDataset
from src_track.evaluation.metrics import MetricsAccumulator, print_metrics
from src_track.models.src_track import SRCTrack, compute_sensitivity_ellipse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="TCND_vn/")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--output_dir",  default="results/")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--compute_sec", action="store_true",
                   help="Compute Sensitivity Ellipse Calibration (slow)")
    p.add_argument("--sec_samples", type=int, default=100,
                   help="How many test sequences to use for SEC")
    return p.parse_args()


@torch.no_grad()
def run_eval(model, loader, device, tag="TEST"):
    model.eval()
    acc = MetricsAccumulator()
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch)
        acc.update(
            out["pred_traj"], batch["gt_traj"],
            regime_labels=batch.get("regime_label"),
            regime_pred=out.get("regime_probs"),
        )
    metrics = acc.compute()
    print_metrics(metrics, tag=tag)
    return metrics


def run_sec(model, dataset, device, n_samples=100):
    """
    Sensitivity Ellipse Calibration:
    Coverage = fraction of test storms where actual_72h_error ≤ major_axis_km.
    Target: ~67% (1-sigma).
    """
    model.eval()
    covered = 0
    total   = 0

    # Estimate sigma_steering from dataset (steering flow variance)
    # Using identity (unit perturbation) as fallback
    sigma_steering = torch.eye(2)

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)),
                               replace=False)
    for idx in indices:
        sample = dataset[idx]
        batch  = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in sample.items()}

        # Ground truth 72h error
        with torch.no_grad():
            out   = model(batch)
            pred  = out["pred_traj"][0, -1]   # [2] lat/lon at 72h
            gt    = sample["gt_traj"][-1]     # [2]
            lat1, lon1 = pred[0].item(), pred[1].item()
            lat2, lon2 = gt[0].item(), gt[1].item()
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a_hav = (math.sin(dlat/2)**2
                     + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
                     * math.sin(dlon/2)**2)
            actual_error_km = 2 * 6371 * math.asin(min(math.sqrt(a_hav), 1.0))

        # Sensitivity ellipse
        ellipse = compute_sensitivity_ellipse(model, batch, sigma_steering)
        major   = ellipse["major_axis_km"]

        if actual_error_km <= major:
            covered += 1
        total += 1

    sec = covered / max(total, 1)
    print(f"\n  SEC (Sensitivity Ellipse Calibration):")
    print(f"    Coverage: {covered}/{total} = {sec:.2%}")
    ok = "OK" if 0.60 <= sec <= 0.75 else ".."
    print(f"    [{ok}] Target: 67% ± (60–75%)")
    if sec < 0.60:
        print("    → sigma_steering underestimated → inflate it")
    elif sec > 0.75:
        print("    → sigma_steering overestimated → deflate it")
    return {"sec": sec, "sec_covered": covered, "sec_total": total}


import math


def main():
    args = parse_args()
    cfg  = get_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nSRC-Track v2 Inference | device={device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    ck     = torch.load(args.checkpoint, map_location="cpu")
    model  = SRCTrack()
    model.load_state_dict(ck["model_state"])
    model  = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {ck.get('epoch','?')})")

    # Test dataset
    test_ds = SRCTrackDataset(
        data1d_path=os.path.join(args.data_root, "Data1d/test"),
        data3d_dir=os.path.join(args.data_root, "Data3d"),
        env_data_dir=os.path.join(args.data_root, "Env_Data"),
        regime_label_csv=os.path.join(args.data_root, "sequence_regime_labels.csv"),
        obs_len=cfg.data.obs_len,
        pred_len=cfg.data.pred_len,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    print(f"Test sequences: {len(test_ds)}")

    # Evaluate
    metrics = run_eval(model, test_loader, device, tag="TEST")

    # SEC (optional, slow)
    if args.compute_sec:
        sec_metrics = run_sec(model, test_ds, device, n_samples=args.sec_samples)
        metrics.update(sec_metrics)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in metrics.items()}, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Comparison table
    ST = {"ADE":224.4,"ATE":213.7,"CTE":59.4,"72h":423.3}
    print("\n  Final comparison:")
    for k, ref in ST.items():
        v = metrics.get(k, float("nan"))
        if not math.isnan(v):
            delta = v - ref
            wins  = "BEAT" if v < ref else "miss"
            print(f"    {k:>4}: SRC={v:.1f}  ST-Trans={ref}  [{wins} {delta:+.1f}km]")


if __name__ == "__main__":
    main()

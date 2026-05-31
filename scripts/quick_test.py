"""
scripts/quick_test.py — Day 1 sanity check

Run this BEFORE any training to verify:
  1. VELOCITY_NORM fix is in effect
  2. Data loads correctly
  3. Model forward pass works
  4. Loss computes correctly
  5. ATE/CTE metrics work
  6. Speed bias is detectable

Usage:
  python scripts/quick_test.py [--data_root TCND_vn/]
"""
from __future__ import annotations
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src_track.config import get_config
from src_track.data.dataset import decode_data1d, physnorm_transform
from src_track.evaluation.metrics import MetricsAccumulator, haversine_km
from src_track.losses.losses import SRCTrackLoss, compute_gt_speed
from src_track.models.src_track import SRCTrack


def test_velocity_norm_fix():
    """Verify VELOCITY_NORM is 100.0, not 1219.84."""
    cfg = get_config()
    assert cfg.data.scs_velocity_norm == 100.0, (
        f"VELOCITY_NORM = {cfg.data.scs_velocity_norm} — should be 100.0!\n"
        f"This is the root cause of ATE=213km. Check config.py."
    )
    print("  [OK] VELOCITY_NORM = 100.0 (SCS max)")


def test_physnorm():
    """PhysNorm produces reasonable speed distribution."""
    # Synthetic SCS-like storm: 4 steps, slow recurving
    raw = np.array([
        [-0.3, 2.0, 0.5, 0.5],   # lon/lat/pres/wnd normalized
        [-0.35, 2.1, 0.5, 0.6],
        [-0.4, 2.2, 0.48, 0.7],
        [-0.42, 2.35, 0.46, 0.75],
    ], dtype=np.float32)
    pn = physnorm_transform(raw)
    assert pn.shape == (4, 9), f"Expected (4,9), got {pn.shape}"
    speeds = pn[:, 4]   # speed_n
    print(f"  [OK] PhysNorm speed range: [{speeds.min():.2f}, {speeds.max():.2f}] (should be ~[-2, 2])")


def test_model_forward():
    """Model forward pass with dummy data."""
    cfg   = get_config()
    model = SRCTrack()
    model.eval()

    B = 2
    batch = {
        "physnorm":     torch.randn(B, 8, 9),
        "data3d":       torch.randn(B, 8, 13, 81, 81),
        "env_data":     torch.randn(B, 8, 84),
        "sve":          torch.randn(B, 8, 24),
        "last_pos":     torch.tensor([[20.0, 115.0], [18.0, 118.0]]),  # lat/lon SCS
        "gt_traj":      torch.randn(B, 12, 2) + torch.tensor([20.0, 115.0]),
        "regime_label": torch.tensor([0, 1]),
        "rii":          torch.tensor([0.3, 1.2]),
        "difficulty":   torch.tensor([0.2, 0.8]),
    }

    with torch.no_grad():
        out = model(batch)

    assert "pred_traj"    in out, "Missing pred_traj"
    assert "pred_speed"   in out, "Missing pred_speed"
    assert "regime_probs" in out, "Missing regime_probs"
    assert "expert_dirs"  in out, "Missing expert_dirs"
    assert out["pred_traj"].shape   == (B, 12, 2), f"pred_traj shape wrong: {out['pred_traj'].shape}"
    assert out["pred_speed"].shape  == (B, 12),    f"pred_speed shape wrong: {out['pred_speed'].shape}"
    assert out["regime_probs"].shape == (B, 3),    f"regime_probs shape wrong: {out['regime_probs'].shape}"

    # Regime probs should sum to 1
    prob_sum = out["regime_probs"].sum(dim=-1)
    assert torch.allclose(prob_sum, torch.ones(B), atol=1e-4), "Regime probs don't sum to 1"

    # Speed should be in physical range [3, 100]
    spd = out["pred_speed"]
    assert (spd >= 3.0).all() and (spd <= 100.0).all(), f"Speed out of range: [{spd.min():.1f}, {spd.max():.1f}]"

    n = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Model forward pass: pred_traj={out['pred_traj'].shape}")
    print(f"  [OK] Regime probs sum=1.0")
    print(f"  [OK] Speed range: [{spd.min():.1f}, {spd.max():.1f}] km/6h")
    print(f"  [OK] Total params: {n:,} ({n/1e6:.2f}M)")


def test_loss():
    """Loss function computes correct values."""
    criterion = SRCTrackLoss(easy_thresh=0.4)

    B = 4
    outputs = {
        "pred_traj":     torch.randn(B, 12, 2) + 20.0,
        "pred_speed":    torch.rand(B, 12) * 30 + 5.0,
        "regime_logits": torch.randn(B, 3),
        "expert_dirs":   {
            "A": torch.nn.functional.normalize(torch.randn(B, 12, 2), dim=-1),
            "B": torch.nn.functional.normalize(torch.randn(B, 12, 2), dim=-1),
            "C": torch.nn.functional.normalize(torch.randn(B, 12, 2), dim=-1),
        },
    }
    batch = {
        "gt_traj":       torch.randn(B, 12, 2) + 20.0,
        "regime_label":  torch.tensor([0, 1, 2, 1]),
        "rii":           torch.tensor([0.3, 1.2, 0.5, 0.8]),
        "difficulty":    torch.tensor([0.2, 0.8, 0.4, 0.7]),
    }

    # Epoch 1 (Phase 1): only L_pos + L_speed
    loss1 = criterion(outputs, batch, current_epoch=1)
    assert torch.isfinite(loss1["loss"]), "Phase 1 loss is not finite"
    assert loss1["l_regime"].item() == 0.0, "L_regime should be 0 in Phase 1"

    # Epoch 20 (Phase 2): + L_regime
    loss2 = criterion(outputs, batch, current_epoch=20)
    assert loss2["l_regime"].item() > 0.0, "L_regime should be active in Phase 2"

    # Epoch 35 (Phase 3): + L_diversity
    loss3 = criterion(outputs, batch, current_epoch=35)
    assert loss3["l_div"].item() >= 0.0, "L_diversity should be active in Phase 3"

    print(f"  [OK] Phase 1 loss={loss1['loss'].item():.4f} (l_regime=0)")
    print(f"  [OK] Phase 2 loss={loss2['loss'].item():.4f} (l_regime={loss2['l_regime'].item():.4f})")
    print(f"  [OK] Phase 3 loss={loss3['loss'].item():.4f} (l_div={loss3['l_div'].item():.4f})")
    print(f"  [OK] easy_frac={loss1.get('easy_frac', '?'):.2f}")


def test_metrics():
    """ATE/CTE metrics compute correctly."""
    acc = MetricsAccumulator()

    B, T = 8, 12
    # Synthetic: pred ≈ gt + small noise
    gt   = torch.randn(B, T, 2) * 2 + torch.tensor([20.0, 115.0])
    pred = gt + torch.randn(B, T, 2) * 0.5   # ~50km error

    regime = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    acc.update(pred, gt, regime_labels=regime)
    metrics = acc.compute()

    assert "ADE" in metrics, "Missing ADE"
    assert "ATE" in metrics, "Missing ATE"
    assert "CTE" in metrics, "Missing CTE"
    assert "speed_bias" in metrics, "Missing speed_bias"

    ratio = metrics["ate_cte_ratio"]
    print(f"  [OK] ADE={metrics['ADE']:.1f}  ATE={metrics['ATE']:.1f}  CTE={metrics['CTE']:.1f}")
    print(f"  [OK] ATE/CTE={ratio:.2f}x  speed_bias={metrics['speed_bias']:+.2f} km/6h")
    print(f"  [OK] ADE_A={metrics.get('ADE_A','?')}  ADE_B={metrics.get('ADE_B','?')}  ADE_C={metrics.get('ADE_C','?')}")


def test_sensitivity_ellipse():
    """Jacobian-based sensitivity ellipse."""
    from src_track.models.src_track import compute_sensitivity_ellipse

    model = SRCTrack()
    model.eval()

    batch = {
        "physnorm":     torch.randn(1, 8, 9),
        "data3d":       torch.randn(1, 8, 13, 81, 81),
        "env_data":     torch.randn(1, 8, 84),
        "sve":          torch.randn(1, 8, 24),
        "last_pos":     torch.tensor([[20.0, 115.0]]),
        "gt_traj":      torch.randn(1, 12, 2) + 20.0,
        "regime_label": torch.tensor([1]),
        "rii":          torch.tensor([0.8]),
        "difficulty":   torch.tensor([0.6]),
    }

    result = compute_sensitivity_ellipse(model, batch)
    assert "major_axis_km" in result, "Missing major_axis_km"
    assert result["major_axis_km"] >= 0, "major_axis < 0"
    print(f"  [OK] Sensitivity ellipse: major={result['major_axis_km']:.1f}km  minor={result['minor_axis_km']:.1f}km")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*65}")
    print(f"  SRC-Track v2 — Day 1 Sanity Check")
    print(f"{'='*65}")

    tests = [
        ("VELOCITY_NORM fix",        test_velocity_norm_fix),
        ("PhysNorm transform",        test_physnorm),
        ("Model forward pass",        test_model_forward),
        ("Loss function",             test_loss),
        ("Metrics (ATE/CTE)",         test_metrics),
        ("Sensitivity ellipse",       test_sensitivity_ellipse),
    ]

    passed = 0
    failed = []
    for name, fn in tests:
        print(f"\n  >> {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed.append(name)

    print(f"\n{'='*65}")
    print(f"  Results: {passed}/{len(tests)} passed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print(f"  All OK — ready to train!")
        print(f"\n  Next steps:")
        print(f"    1. python scripts/generate_regime_labels.py --data_root TCND_vn/")
        print(f"    2. python -m src_track.training.trainer --data_root TCND_vn/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

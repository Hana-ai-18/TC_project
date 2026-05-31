"""
SRC-Track v2 — Training Loop
═══════════════════════════════════════════════════════════════

NEW vs v1:
  [FIX-1]  VELOCITY_NORM 1219.84 → 100.0 via config (1-line fix, biggest impact)
  [FIX-2]  Adaptive threshold: keeps easy_frac ≈ 50% (from v74 train script)
  [FIX-3]  Easy/Hard curriculum as per thầy's suggestion
  [FIX-4]  ATE/CTE logged every eval (was missing from v1)
  [FIX-5]  Regime-conditioned difficulty weighting in L_pos

Curriculum:
  Phase 1 (ep 1–15)  : Easy only (diff < 0.4), L_pos + L_speed
  Phase 2 (ep 16–30) : Easy+Medium (diff < 0.7), + L_regime
  Phase 3 (ep 31–50) : All samples, + L_diversity, focal Regime B 1.5×
  Phase 4 (ep 51–70) : Fine-tune, LR × 0.1

Monitor targets:
  ATE/CTE ratio  < 2.5×  (was 3.58× with VELOCITY_NORM bug)
  speed_bias     < 3 km/6h
  regime_acc     > 70%   (from epoch 16+)
  expert_cos_sim < 0.3   (Regime B sequences, from epoch 31+)
  val ADE        < 170 km

Usage:
  python -m src_track.training.trainer --help
  python -m src_track.training.trainer \\
      --data_root   TCND_vn/ \\
      --save_dir    checkpoints/v2/ \\
      --max_epochs  70 \\
      --batch_size  32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src_track.config import SRCTrackConfig, get_config
from src_track.data.dataset import SRCTrackDataset
from src_track.losses.losses import SRCTrackLoss
from src_track.models.src_track import SRCTrack, build_model
from src_track.evaluation.metrics import MetricsAccumulator, print_metrics


# ─────────────────────────────────────────────────────────────
# Baseline references (ST-Trans trained on same SCS test set)
# ─────────────────────────────────────────────────────────────
ST_TRANS = {
    "ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
    "12h": 77.5,  "24h": 130.5, "48h": 269.9, "72h": 423.3,
}
LSTM_REF = {"ADE": 245.3, "ATE": 235.7, "CTE": 67.3, "72h": 461.9}
FM_REF   = {"ADE": 236.1, "ATE": 223.6, "CTE": 74.9, "72h": 471.6}


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict, device: str) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# [FIX-2] Adaptive Threshold (from v74 train script)
# Keeps easy_frac ≈ 50% regardless of dataset distribution
# ─────────────────────────────────────────────────────────────

class AdaptiveThreshold:
    """
    EMA of batch median difficulty → threshold that always splits ~50/50.

    Problem v1 had: fixed easy_thresh=0.4 → with TC dataset, many batches
    end up 80%+ "hard" (difficulty scores cluster around 0.5–0.7), causing
    L_hard to dominate and making epoch-0 loss very large.

    Fix: EMA of running batch median → always ~50% easy, ~50% hard.
    """
    def __init__(self, init_thresh: float = 0.35, ema_alpha: float = 0.02,
                 clip_min: float = 0.15, clip_max: float = 0.65):
        self.thresh    = init_thresh
        self._ema      = init_thresh
        self.alpha     = ema_alpha
        self.clip_min  = clip_min
        self.clip_max  = clip_max

    def update(self, diff_vals: torch.Tensor) -> float:
        """diff_vals: [B] tensor of difficulty scores."""
        batch_median = float(diff_vals.median().item())
        self._ema = (1. - self.alpha) * self._ema + self.alpha * batch_median
        self.thresh = float(np.clip(self._ema, self.clip_min, self.clip_max))
        return self.thresh

    def get(self) -> float:
        return self.thresh


# ─────────────────────────────────────────────────────────────
# Curriculum helpers
# ─────────────────────────────────────────────────────────────

def get_difficulty_max(epoch: int, cfg: SRCTrackConfig) -> Optional[float]:
    t = cfg.train
    if epoch <= t.phase1_end:
        return t.phase1_diff_max   # 0.4
    elif epoch <= t.phase2_end:
        return t.phase2_diff_max   # 0.7
    else:
        return None                # all samples


def make_sampler(dataset: SRCTrackDataset, epoch: int,
                 cfg: SRCTrackConfig) -> Optional[WeightedRandomSampler]:
    """Phase 3+: oversample Regime B by 1.5×."""
    if epoch <= cfg.train.phase2_end:
        return None
    weights = [1.5 if s['regime'] == 1 else 1.0 for s in dataset.sequences]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_dataloader(dataset: SRCTrackDataset, epoch: int,
                    cfg: SRCTrackConfig, shuffle: bool = True) -> DataLoader:
    sampler = make_sampler(dataset, epoch, cfg) if shuffle else None
    return DataLoader(
        dataset,
        batch_size  = cfg.train.batch_size,
        sampler     = sampler,
        shuffle     = (shuffle and sampler is None),
        num_workers = cfg.train.num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )


# ─────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────

def save_checkpoint(path: str, epoch: int, model: SRCTrack,
                    optimizer, scheduler, metrics: Dict, cfg: SRCTrackConfig):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "metrics":     metrics,
        "config":      cfg,
    }, path)


def load_checkpoint(path: str, model: SRCTrack, optimizer=None, scheduler=None):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    if optimizer and "optim_state" in ck:
        try:
            optimizer.load_state_dict(ck["optim_state"])
        except Exception as e:
            print(f"  [warn] optimizer state not loaded: {e}")
    if scheduler and "sched_state" in ck:
        try:
            scheduler.load_state_dict(ck["sched_state"])
        except Exception:
            pass
    return ck.get("epoch", 0), ck.get("metrics", {})


# ─────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     SRCTrack,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SRCTrackLoss,
    epoch:     int,
    cfg:       SRCTrackConfig,
    adap:      Optional[AdaptiveThreshold] = None,
    scaler=None,
) -> Dict[str, float]:

    model.train()
    device = cfg.train.device
    total_loss = 0.0
    loss_terms = defaultdict(float)
    n_batches  = 0
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        batch = move_batch(batch, device)

        # [FIX-2] Adaptive threshold update
        if adap is not None and cfg.train.use_adaptive_thresh:
            new_thresh = adap.update(batch["difficulty"])
            criterion.easy_thresh = new_thresh

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                losses  = criterion(outputs, batch, epoch)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch)
            losses  = criterion(outputs, batch, epoch)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        total_loss += losses["loss"].item()
        for k, v in losses.items():
            if k != "loss":
                loss_terms[k] += v.item() if hasattr(v, "item") else float(v)
        n_batches += 1

        # Log every 20 batches
        if i % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            thresh_str = f" thr={adap.get():.2f}" if adap else ""
            ef_str = ""
            if "easy_frac" in loss_terms:
                ef = loss_terms["easy_frac"] / max(n_batches, 1)
                ef_str = f" efrac={ef:.2f}"
            print(
                f"  [{epoch:>3}][{i:>4}/{len(loader)}] "
                f"loss={losses['loss'].item():.3f} "
                f"pos={losses.get('l_pos', 0.):.3f} "
                f"spd={losses.get('l_speed', 0.):.3f} "
                f"reg={losses.get('l_regime', 0.):.3f} "
                f"div={losses.get('l_div', 0.):.3f}"
                f"{ef_str}{thresh_str} lr={lr:.1e}"
            )

    elapsed = time.perf_counter() - t0
    return {
        "loss":   total_loss / max(n_batches, 1),
        "elapsed": elapsed,
        **{k: v / max(n_batches, 1) for k, v in loss_terms.items()},
    }


# ─────────────────────────────────────────────────────────────
# [FIX-4] Evaluation with ATE/CTE + regime breakdown
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:  SRCTrack,
    loader: DataLoader,
    device: str,
    tag:    str = "",
) -> Dict[str, float]:
    model.eval()
    acc = MetricsAccumulator()
    t0  = time.perf_counter()

    for batch in loader:
        batch  = move_batch(batch, device)
        out    = model(batch)
        pred   = out["pred_traj"]   # [B, T, 2] lat°, lon°
        gt     = batch["gt_traj"]

        # Per-sample regime for RCE breakdown
        regime = batch.get("regime_label")
        rii    = batch.get("rii")

        acc.update(pred, gt,
                   regime_labels=regime,
                   rii=rii,
                   regime_pred=out.get("regime_probs"))

    results = acc.compute()
    elapsed = time.perf_counter() - t0

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  [{tag}  {elapsed:.0f}s]")
    for k in ["ADE", "12h", "24h", "48h", "72h"]:
        v = results.get(k, float("nan"))
        ref = ST_TRANS.get(k, float("nan"))
        ok = "✅" if v < ref else "❌"
        print(f"  {k:>4}={v:>7.1f}km {ok} ST-Trans={ref}")
    if "ATE" in results and "CTE" in results:
        ate, cte = results["ATE"], results["CTE"]
        ratio = ate / max(cte, 1e-6)
        print(f"  ATE={ate:.1f}  CTE={cte:.1f}  ratio={ratio:.2f}x "
              f"(target <2.5x, was 3.58x)")
    if "speed_bias" in results:
        print(f"  speed_bias={results['speed_bias']:+.2f} km/6h (target <3)")
    if "regime_acc" in results:
        print(f"  regime_acc={results['regime_acc']:.2%} (target >70%)")

    # Per-regime ADE
    for r, rname in [(0, "A"), (1, "B"), (2, "C")]:
        k = f"ADE_{rname}"
        if k in results:
            print(f"  ADE_{rname}={results[k]:.1f}")

    print(f"{'='*70}\n")
    return results


# ─────────────────────────────────────────────────────────────
# Best model saver
# ─────────────────────────────────────────────────────────────

class BestModelSaver:
    def __init__(self, save_dir: str, patience: int = 15, min_epoch: int = 30):
        self.save_dir  = save_dir
        self.patience  = patience
        self.min_epoch = min_epoch
        self.best_ade  = float("inf")
        self.best_72h  = float("inf")
        self.best_ate  = float("inf")
        self.best_cte  = float("inf")
        self.no_improve = 0
        self.stop      = False
        os.makedirs(save_dir, exist_ok=True)

    def update(self, metrics: Dict, epoch: int, model: SRCTrack,
               optimizer, scheduler, cfg: SRCTrackConfig,
               train_loss: float, val_loss: float) -> bool:
        ade = metrics.get("ADE", float("inf"))
        h72 = metrics.get("72h", float("inf"))
        ate = metrics.get("ATE", float("inf"))
        cte = metrics.get("CTE", float("inf"))

        improved = False
        if ade < self.best_ade:
            self.best_ade = ade
            save_checkpoint(
                os.path.join(self.save_dir, "best_ade.pth"),
                epoch, model, optimizer, scheduler,
                {"ADE": ade, "72h": h72, "ATE": ate, "CTE": cte}, cfg
            )
            print(f"  [BEST ADE] ep={epoch} ADE={ade:.1f}")
            improved = True
        if h72 < self.best_72h:
            self.best_72h = h72
            save_checkpoint(
                os.path.join(self.save_dir, "best_72h.pth"),
                epoch, model, optimizer, scheduler,
                {"ADE": ade, "72h": h72, "ATE": ate, "CTE": cte}, cfg
            )
            print(f"  [BEST 72h] ep={epoch} 72h={h72:.1f}")
            improved = True
        if ate < self.best_ate:
            self.best_ate = ate
            save_checkpoint(
                os.path.join(self.save_dir, "best_ate.pth"),
                epoch, model, optimizer, scheduler,
                {"ADE": ade, "72h": h72, "ATE": ate, "CTE": cte}, cfg
            )
            improved = True
        if cte < self.best_cte:
            self.best_cte = cte
            save_checkpoint(
                os.path.join(self.save_dir, "best_cte.pth"),
                epoch, model, optimizer, scheduler,
                {"ADE": ade, "72h": h72, "ATE": ate, "CTE": cte}, cfg
            )
            improved = True

        if improved:
            self.no_improve = 0
        else:
            self.no_improve += 1
            if epoch >= self.min_epoch and self.no_improve >= self.patience:
                self.stop = True
                print(f"  [EARLY STOP] ep={epoch}")

        return improved


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────

def train(cfg: SRCTrackConfig, args=None):
    set_seed(cfg.train.seed)
    device = cfg.train.device
    print(f"\n{'='*72}")
    print(f"  SRC-Track v2 — Steering Regime-Conditioned TC Track")
    print(f"  Device: {device}")
    print(f"  [FIX] VELOCITY_NORM: 1219.84 → {cfg.data.scs_velocity_norm} (SCS max)")
    print(f"  [NEW] Easy/Hard curriculum with adaptive threshold")
    print(f"  [NEW] 3-class Regime Classifier + MoE Decoder")
    print(f"  [NEW] Physics Jacobian Sensitivity Ellipse (inference)")
    print(f"\n  Targets vs ST-Trans (ADE=224.4, ATE=213.7, CTE=59.4, 72h=423.3):")
    print(f"    ADE < 170 km | ATE < 150 | CTE < 55 | 72h < 350")
    print(f"    ATE/CTE ratio < 2.5× (was 3.58× with normalization bug)")
    print(f"{'='*72}\n")

    # ── Datasets ──────────────────────────────────────────────
    diff_max = get_difficulty_max(1, cfg)
    train_ds = SRCTrackDataset(
        data1d_path=cfg.data.data1d_train,
        data3d_dir=cfg.data.data3d_dir,
        env_data_dir=cfg.data.env_data_dir,
        regime_label_csv=cfg.data.regime_label_csv,
        obs_len=cfg.data.obs_len,
        pred_len=cfg.data.pred_len,
        stride=cfg.data.stride,
        speed_mean=cfg.data.scs_speed_mean,
        speed_std=cfg.data.scs_speed_std,
        max_difficulty=diff_max,
    )
    val_ds = SRCTrackDataset(
        data1d_path=cfg.data.data1d_val,
        data3d_dir=cfg.data.data3d_dir,
        env_data_dir=cfg.data.env_data_dir,
        regime_label_csv=cfg.data.regime_label_csv,
        obs_len=cfg.data.obs_len,
        pred_len=cfg.data.pred_len,
        stride=cfg.data.stride,
        speed_mean=cfg.data.scs_speed_mean,
        speed_std=cfg.data.scs_speed_std,
        max_difficulty=None,
    )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    print(f"  train: {len(train_ds)} seqs | val: {len(val_ds)} seqs")

    # ── Model ─────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = count_params(model)
    print(f"  Params: {n_params:,}  ({n_params/1e6:.2f}M)")

    # ── Optimizer + Scheduler ─────────────────────────────────
    optimizer = AdamW(model.parameters(),
                      lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train.max_epochs, eta_min=cfg.train.min_lr
    )

    # AMP scaler (optional)
    scaler = torch.cuda.amp.GradScaler() if (
        torch.cuda.is_available() and getattr(args, "use_amp", False)
    ) else None

    # ── Loss ──────────────────────────────────────────────────
    criterion = SRCTrackLoss(
        w_speed=cfg.loss.w_speed,
        w_regime=cfg.loss.w_regime,
        w_div=cfg.loss.w_div,
        regime_start_epoch=cfg.loss.regime_start_epoch,
        div_start_epoch=cfg.loss.div_start_epoch,
        huber_delta=cfg.loss.huber_delta,
        step_w_min=cfg.loss.step_w_min,
        step_w_max=cfg.loss.step_w_max,
        rii_threshold=cfg.loss.rii_threshold,
        regime_b_extra=cfg.loss.regime_b_extra,
    )

    # ── [FIX-2] Adaptive threshold ───────────────────────────
    adap = AdaptiveThreshold(
        init_thresh=cfg.train.easy_thresh_init,
        ema_alpha=cfg.train.adaptive_ema_alpha,
    ) if cfg.train.use_adaptive_thresh else None

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 1
    resume_path = getattr(args, "resume", None)
    if resume_path and os.path.exists(resume_path):
        start_epoch, prev_metrics = load_checkpoint(
            resume_path, model, optimizer, scheduler)
        start_epoch += 1
        print(f"  Resumed from {resume_path} (epoch {start_epoch})")

    # ── Saver ─────────────────────────────────────────────────
    saver = BestModelSaver(
        cfg.train.save_dir,
        patience=cfg.train.patience,
        min_epoch=cfg.train.phase2_end,
    )

    # ── Training loop ─────────────────────────────────────────
    history = []
    for epoch in range(start_epoch, cfg.train.max_epochs + 1):
        print(f"\n  ── Epoch {epoch}/{cfg.train.max_epochs} ─────────────")

        # Phase transitions
        phase = ("1-warmup" if epoch <= cfg.train.phase1_end else
                  "2-regime" if epoch <= cfg.train.phase2_end else
                  "3-experts" if epoch <= cfg.train.phase3_end else
                  "4-finetune")
        print(f"     Phase {phase}")

        # Phase 4: reduce LR
        if epoch == cfg.train.phase3_end + 1:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1
            print(f"  [Phase 4] LR reduced by 10×")

        # Update dataset difficulty filter
        diff_max = get_difficulty_max(epoch, cfg)
        if diff_max != train_ds.max_difficulty:
            train_ds.max_difficulty = diff_max
            train_ds.sequences = [s for s in train_ds._all_sequences
                                   if diff_max is None or s["difficulty"] <= diff_max]
            print(f"  [Curriculum] max_diff={diff_max}, sequences={len(train_ds)}")

        # Make dataloader (may have regime B oversampling)
        train_loader = make_dataloader(train_ds, epoch, cfg, shuffle=True)

        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, criterion,
            epoch, cfg, adap=adap, scaler=scaler,
        )

        # Val loss (fast)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vb in val_loader:
                vb = move_batch(vb, device)
                vo = model(vb)
                vl = criterion(vo, vb, epoch)
                val_loss += vl["loss"].item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step()

        print(f"  train_loss={train_stats['loss']:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"thresh={adap.get():.3f}" if adap else
              f"  train_loss={train_stats['loss']:.4f}  val_loss={val_loss:.4f}")

        # Full evaluation every 3 epochs
        if epoch % 3 == 0 or epoch == cfg.train.max_epochs:
            metrics = evaluate(model, val_loader, device,
                               tag=f"VAL ep{epoch}")
            saver.update(metrics, epoch, model, optimizer, scheduler,
                         cfg, train_stats["loss"], val_loss)

            row = {"epoch": epoch, "train_loss": train_stats["loss"],
                   "val_loss": val_loss, **metrics}
            history.append(row)

        # Periodic checkpoint
        if epoch % cfg.train.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.train.save_dir, f"ckpt_ep{epoch:03d}.pth"),
                epoch, model, optimizer, scheduler, {}, cfg,
            )

        if saver.stop:
            print(f"  Early stop at epoch {epoch}.")
            break

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best ADE: {saver.best_ade:.1f} km (ST-Trans: {ST_TRANS['ADE']})")
    print(f"  Best 72h: {saver.best_72h:.1f} km (ST-Trans: {ST_TRANS['72h']})")
    print(f"  Best ATE: {saver.best_ate:.1f} km (ST-Trans: {ST_TRANS['ATE']})")
    print(f"  Best CTE: {saver.best_cte:.1f} km (ST-Trans: {ST_TRANS['CTE']})")
    # Save training history
    hist_path = os.path.join(cfg.train.save_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {hist_path}")
    print(f"{'='*72}\n")

    return saver.best_ade


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SRC-Track v2 — SCS TC Track Forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root",   default="TCND_vn",
                   help="Root of TCND_vn dataset (contains Data1d/, Data3d/, Env_Data/)")
    p.add_argument("--save_dir",    default="checkpoints/v2/")
    p.add_argument("--max_epochs",  type=int,   default=70)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--use_amp",     action="store_true",
                   help="Enable AMP (mixed precision, requires CUDA)")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_adaptive_thresh", action="store_true",
                   help="Disable adaptive threshold (use fixed 0.4)")
    p.add_argument("--seed",        type=int, default=42)
    # Aliases for backward compat with old Kaggle notebooks
    p.add_argument("--dataset_root",  default=None,
                   help="Alias for --data_root")
    p.add_argument("--output_dir",    default=None,
                   help="Alias for --save_dir")
    p.add_argument("--learning_rate", type=float, default=None,
                   help="Alias for --lr")
    p.add_argument("--num_epochs",    type=int,   default=None,
                   help="Alias for --max_epochs")
    p.add_argument("--regime_csv",    default=None,
                   help="Explicit path to regime_labels.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args = _apply_aliases(args)
    cfg  = get_config()

    # Apply CLI overrides
    cfg.data.data1d_train = os.path.join(args.data_root, "Data1d/train")
    cfg.data.data1d_val   = os.path.join(args.data_root, "Data1d/val")
    cfg.data.data1d_test  = os.path.join(args.data_root, "Data1d/test")
    cfg.data.data3d_dir   = os.path.join(args.data_root, "Data3d")
    cfg.data.env_data_dir = os.path.join(args.data_root, "Env_Data")
    cfg.train.save_dir    = args.save_dir
    cfg.train.max_epochs  = args.max_epochs
    cfg.train.batch_size  = args.batch_size
    cfg.train.lr          = args.lr
    cfg.train.num_workers = args.num_workers
    cfg.train.seed        = args.seed
    if args.no_adaptive_thresh:
        cfg.train.use_adaptive_thresh = False
    if hasattr(args, 'regime_csv') and args.regime_csv:
        cfg.data.regime_label_csv = args.regime_csv

    train(cfg, args)

# ─────────────────────────────────────────────────────────────
# Backward-compat aliases (Kaggle notebook style)
# ─────────────────────────────────────────────────────────────
def _apply_aliases(args):
    """Map old-style arg names → new names so both work."""
    # --dataset_root → --data_root
    if hasattr(args, 'dataset_root') and args.dataset_root is not None:
        if not hasattr(args, 'data_root') or args.data_root == 'TCND_vn':
            args.data_root = args.dataset_root
    # --output_dir → --save_dir
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        if not hasattr(args, 'save_dir') or args.save_dir == 'checkpoints/v2/':
            args.save_dir = args.output_dir
    # --learning_rate → --lr
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        args.lr = args.learning_rate
    # --num_epochs → --max_epochs
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        args.max_epochs = args.num_epochs
    return args

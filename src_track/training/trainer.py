"""
SRC-Track v2 — Training Loop (FIXED)

BUGS FIXED vs uploaded version:
  BUG-1: criterion used before defined → moved SRCTrackLoss() before optimizer
  BUG-2: optimizer không include criterion.parameters() đúng cách →
         model + criterion params được pass đúng vào AdamW
  BUG-3: val_loss print logic sai (ternary trả về string thay vì in) → fix print
  BUG-4: _apply_aliases() được gọi TRƯỚC khi định nghĩa → move lên trên main
  BUG-5: fast val (mỗi epoch) không in metrics → thêm print_metrics() cho fast eval
  BUG-6: full val chỉ in khi epoch%3==0, fast val không in gì → thêm fast val metrics
  BUG-7: SRCTrackLoss.__init__() thiếu tham số step_w_min/step_w_max so với class signature
  BUG-8: make_sampler yêu cầu SRCTrackDataset nhưng nhận Subset → fix type check

Kaggle usage:
  !python scripts/train_src.py \
      --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
      --output_dir   /kaggle/working/runs/src_v1 \
      --batch_size 32 --num_epochs 70 --learning_rate 1e-4 --use_amp
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
from typing import Dict, Optional

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

ST_TRANS = {
    "ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
    "12h": 77.5,  "24h": 130.5, "48h": 269.9, "72h": 423.3,
}
FM_REF  = {"ADE": 236.1, "ATE": 223.6, "CTE": 74.9, "72h": 471.6}


# ─── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict, device: str) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── BUG-4 FIX: _apply_aliases defined BEFORE main ───────────────────────────

def _apply_aliases(args):
    """Map Kaggle-style arg names → canonical names."""
    if getattr(args, 'dataset_root', None):
        if not getattr(args, 'data_root', None) or args.data_root == 'TCND_vn':
            args.data_root = args.dataset_root
    if getattr(args, 'output_dir', None):
        if not getattr(args, 'save_dir', None) or args.save_dir == 'checkpoints/v2/':
            args.save_dir = args.output_dir
    if getattr(args, 'learning_rate', None):
        args.lr = args.learning_rate
    if getattr(args, 'num_epochs', None):
        args.max_epochs = args.num_epochs
    return args


# ─── AdaptiveThreshold ────────────────────────────────────────────────────────

class AdaptiveThreshold:
    def __init__(self, init_thresh: float = 0.35, ema_alpha: float = 0.02,
                 clip_min: float = 0.15, clip_max: float = 0.65):
        self.thresh = init_thresh; self._ema = init_thresh
        self.alpha = ema_alpha; self.clip_min = clip_min; self.clip_max = clip_max

    def update(self, diff_vals: torch.Tensor) -> float:
        m = float(diff_vals.median().item())
        self._ema = (1. - self.alpha) * self._ema + self.alpha * m
        self.thresh = float(np.clip(self._ema, self.clip_min, self.clip_max))
        return self.thresh

    def get(self) -> float:
        return self.thresh


# ─── Curriculum helpers ───────────────────────────────────────────────────────

def get_difficulty_max(epoch: int, cfg: SRCTrackConfig) -> Optional[float]:
    t = cfg.train
    if epoch <= t.phase1_end:   return t.phase1_diff_max
    elif epoch <= t.phase2_end: return t.phase2_diff_max
    else:                       return None


def make_sampler(dataset, epoch: int,
                 cfg: SRCTrackConfig) -> Optional[WeightedRandomSampler]:
    """Phase 3+: oversample Regime B 1.5×. Works with SRCTrackDataset or Subset."""
    if epoch <= cfg.train.phase2_end:
        return None
    # BUG-8 FIX: get sequences from underlying dataset if Subset
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        base    = dataset.dataset
        indices = dataset.indices
        seqs    = [base.sequences[i] for i in indices]
    else:
        seqs = dataset.sequences
    weights = [1.5 if s['regime'] == 1 else 1.0 for s in seqs]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_dataloader(dataset, epoch: int,
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


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, optimizer, scheduler, metrics, cfg):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    if optimizer and "optim_state" in ck:
        try: optimizer.load_state_dict(ck["optim_state"])
        except Exception as e: print(f"  [warn] optimizer state not loaded: {e}")
    if scheduler and "sched_state" in ck:
        try: scheduler.load_state_dict(ck["sched_state"])
        except: pass
    return ck.get("epoch", 0), ck.get("metrics", {})


# ─── Evaluate — print đầy đủ metrics ─────────────────────────────────────────

@torch.no_grad()
def evaluate(model: SRCTrack, loader: DataLoader,
             device: str, tag: str = "") -> Dict[str, float]:
    model.eval()
    acc = MetricsAccumulator()
    t0  = time.perf_counter()

    for batch in loader:
        batch = move_batch(batch, device)
        out   = model(batch)
        acc.update(
            out["pred_traj"],
            batch["gt_traj"],
            regime_labels=batch.get("regime_label"),
            rii=batch.get("rii"),
            regime_pred=out.get("regime_probs"),
        )

    results = acc.compute()
    elapsed = time.perf_counter() - t0

    # ── Print full metrics table ──────────────────────────────
    print(f"\n{'='*70}")
    print(f"  [{tag}  {elapsed:.0f}s]")

    # Primary metrics vs ST-Trans
    for k in ["ADE", "12h", "24h", "48h", "72h"]:
        v   = results.get(k, float("nan"))
        ref = ST_TRANS.get(k, float("nan"))
        ok  = "✅ BEAT" if v < ref else "❌"
        gap = v - ref
        print(f"  {k:>4} = {v:>7.1f} km  {ok} ST-Trans={ref:.1f}  ({gap:+.1f})")

    # ATE / CTE / ratio
    if "ATE" in results and "CTE" in results:
        ate   = results["ATE"]; cte = results["CTE"]
        ratio = ate / max(cte, 1e-6)
        ok_r  = "✅" if ratio < 2.5 else "❌"
        ok_a  = "✅ BEAT" if ate < ST_TRANS["ATE"] else "❌"
        ok_c  = "✅ BEAT" if cte < ST_TRANS["CTE"] else "❌"
        print(f"  ATE  = {ate:>7.1f} km  {ok_a} ST-Trans={ST_TRANS['ATE']:.1f}")
        print(f"  CTE  = {cte:>7.1f} km  {ok_c} ST-Trans={ST_TRANS['CTE']:.1f}")
        print(f"  ATE/CTE ratio = {ratio:.2f}×  {ok_r} (target <2.5×, was 3.58× with bug)")

    # Speed bias
    if "speed_bias" in results:
        sb = results["speed_bias"]
        ok = "✅" if abs(sb) < 3 else "❌"
        print(f"  speed_bias = {sb:+.2f} km/6h  {ok} (target |bias|<3)")
        print(f"  pred_speed = {results.get('mean_pred_speed', float('nan')):.1f} km/6h  "
              f"gt_speed = {results.get('mean_gt_speed', float('nan')):.1f} km/6h")

    # Regime accuracy
    if "regime_acc" in results:
        ra = results["regime_acc"]
        ok = "✅" if ra > 0.70 else "❌"
        print(f"  regime_acc = {ra:.2%}  {ok} (target >70%)")

    # Per-regime ADE
    regime_line = ""
    for r, rname in [(0,"A"),(1,"B"),(2,"C")]:
        k = f"ADE_{rname}"
        if k in results:
            regime_line += f"  ADE_{rname}={results[k]:.1f}"
    if regime_line:
        print(regime_line)

    # FDE
    if "FDE" in results:
        print(f"  FDE  = {results['FDE']:.1f} km")

    print(f"{'='*70}\n")
    return results


# ─── Train one epoch ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, epoch, cfg, adap=None, scaler=None):
    model.train(); device = cfg.train.device
    total_loss = 0.; loss_terms = defaultdict(float); n = 0
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        batch = move_batch(batch, device)

        if adap is not None and cfg.train.use_adaptive_thresh:
            criterion.easy_thresh = adap.update(batch["difficulty"])

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(batch)
                losses  = criterion(outputs, batch, epoch)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            outputs = model(batch)
            losses  = criterion(outputs, batch, epoch)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        if not torch.isfinite(losses["loss"]):
            if i < 5: print(f"  [WARN] NaN/Inf loss at batch {i}")
            continue

        total_loss += losses["loss"].item()
        for k, v in losses.items():
            if k != "loss":
                loss_terms[k] += float(v) if not hasattr(v,"item") else v.item()
        n += 1

        if i % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            sw_72 = losses.get("sw_sw_72h", losses.get("sw_72h", 0.))
            sw_r  = losses.get("sw_sw_ratio", losses.get("sw_ratio", 0.))
            thr_s = f" thr={adap.get():.2f}" if adap else ""
            print(
                f"  [{epoch:>3}][{i:>4}/{len(loader)}]"
                f" loss={losses['loss'].item():.3f}"
                f" easy={float(losses.get('L_easy', losses.get('l_easy', 0.))):.3f}"
                f" hard={float(losses.get('L_hard', losses.get('l_hard', 0.))):.3f}"
                f" spd={float(losses.get('l_speed', 0.)):.3f}"
                f" reg={float(losses.get('l_regime', 0.)):.3f}"
                f" efrac={float(losses.get('easy_frac', 0.)):.2f}"
                f" sw72={float(sw_72):.2f} swr={float(sw_r):.2f}"
                f"{thr_s} lr={lr:.1e}"
            )

    elapsed = time.perf_counter() - t0
    return {"loss": total_loss / max(n, 1), "elapsed": elapsed,
            **{k: v / max(n,1) for k, v in loss_terms.items()}}


# ─── Best saver ───────────────────────────────────────────────────────────────

class BestModelSaver:
    def __init__(self, save_dir, patience=15, min_epoch=30):
        self.save_dir = save_dir; self.patience = patience; self.min_epoch = min_epoch
        self.best_ade = self.best_72h = self.best_ate = self.best_cte = float("inf")
        self.no_improve = 0; self.stop = False
        os.makedirs(save_dir, exist_ok=True)

    def update(self, metrics, epoch, model, optimizer, scheduler, cfg, tl, vl):
        ade = metrics.get("ADE", float("inf"))
        h72 = metrics.get("72h", float("inf"))
        ate = metrics.get("ATE", float("inf"))
        cte = metrics.get("CTE", float("inf"))
        improved = False
        for v, attr, fname in [(ade,"best_ade","best_ade.pth"),
                                (h72,"best_72h","best_72h.pth"),
                                (ate,"best_ate","best_ate.pth"),
                                (cte,"best_cte","best_cte.pth")]:
            if v < getattr(self, attr):
                setattr(self, attr, v)
                save_checkpoint(os.path.join(self.save_dir, fname),
                                epoch, model, optimizer, scheduler, metrics, cfg)
                print(f"  [BEST {attr.upper()}] ep={epoch}  "
                      f"ADE={ade:.1f}  72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}")
                improved = True
        if improved: self.no_improve = 0
        else:
            self.no_improve += 1
            if epoch >= self.min_epoch and self.no_improve >= self.patience:
                self.stop = True; print(f"  [EARLY STOP] ep={epoch}")
        return improved


# ─── Main training loop ───────────────────────────────────────────────────────

def train(cfg: SRCTrackConfig, args=None):
    set_seed(cfg.train.seed)
    device = cfg.train.device

    print(f"\n{'='*72}")
    print(f"  SRC-Track v2 — Steering Regime-Conditioned TC Track")
    print(f"  Device: {device}")
    print(f"  Baselines: FM={FM_REF['ADE']:.1f}  ST-Trans={ST_TRANS['ADE']:.1f}")
    print(f"  Targets:   ADE<170  ATE<150  CTE<55  72h<350")
    print(f"{'='*72}\n")

    # ── Datasets ──────────────────────────────────────────────
    diff_max = get_difficulty_max(1, cfg)
    train_ds = SRCTrackDataset(
        data1d_path=cfg.data.data1d_train, data3d_dir=cfg.data.data3d_dir,
        env_data_dir=cfg.data.env_data_dir, regime_label_csv=cfg.data.regime_label_csv,
        obs_len=cfg.data.obs_len, pred_len=cfg.data.pred_len, stride=cfg.data.stride,
        speed_mean=cfg.data.scs_speed_mean, speed_std=cfg.data.scs_speed_std,
        max_difficulty=diff_max,
        use_flip_aug=True, use_noise_aug=True, use_intensity_aug=True,
        is_val=False,  # augmentation ON for train
    )
    val_ds = SRCTrackDataset(
        data1d_path=cfg.data.data1d_val, data3d_dir=cfg.data.data3d_dir,
        env_data_dir=cfg.data.env_data_dir, regime_label_csv=cfg.data.regime_label_csv,
        obs_len=cfg.data.obs_len, pred_len=cfg.data.pred_len, stride=cfg.data.stride,
        speed_mean=cfg.data.scs_speed_mean, speed_std=cfg.data.scs_speed_std,
        max_difficulty=None,
        use_flip_aug=False, use_noise_aug=False, use_intensity_aug=False,
        is_val=True,   # no augmentation for val
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers,
                            pin_memory=torch.cuda.is_available(), drop_last=False)
    print(f"  train: {len(train_ds)} seqs  |  val: {len(val_ds)} seqs")

    # ── Model ─────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    print(f"  Params: {count_params(model):,}")

    # BUG-1 FIX: criterion defined BEFORE optimizer
    criterion = SRCTrackLoss(
        w_regime=cfg.loss.w_regime,
        w_div=cfg.loss.w_div,
        regime_start_epoch=cfg.loss.regime_start_epoch,
        div_start_epoch=cfg.loss.div_start_epoch,
        huber_delta=cfg.loss.huber_delta,
        rii_threshold=cfg.loss.rii_threshold,
        rii_weight_scale=cfg.loss.rii_weight_scale,
        regime_b_extra=cfg.loss.regime_b_extra,
        easy_thresh=cfg.train.easy_thresh_init,
        # v3 explicit speed weight
        w_speed=cfg.loss.w_speed,
    ).to(device)

    # FIX: Bỏ warmup, dùng lr constant + CosineAnnealing (đã chứng minh ổn định)
    # Warmup cũ: initial_lr=3e-4 quá cao → oscillation (ADE lên xuống ep3-6)
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = AdamW(all_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    scheduler = CosineAnnealingLR(optimizer,
                                   T_max=cfg.train.max_epochs,
                                   eta_min=cfg.train.min_lr)

    scaler = torch.amp.GradScaler("cuda") if (
        torch.cuda.is_available() and getattr(args, "use_amp", False)) else None

    adap = AdaptiveThreshold(
        init_thresh=cfg.train.easy_thresh_init,
        ema_alpha=cfg.train.adaptive_ema_alpha,
    ) if cfg.train.use_adaptive_thresh else None

    # Resume
    start_epoch = 1
    if getattr(args, "resume", None) and os.path.exists(args.resume):
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
        print(f"  Resumed from {args.resume} (epoch {start_epoch})")

    saver   = BestModelSaver(cfg.train.save_dir, patience=cfg.train.patience,
                              min_epoch=cfg.train.phase2_end)
    history = []

    for epoch in range(start_epoch, cfg.train.max_epochs + 1):
        phase = ("1-warmup"  if epoch <= cfg.train.phase1_end else
                 "2-regime"  if epoch <= cfg.train.phase2_end else
                 "3-experts" if epoch <= cfg.train.phase3_end else "4-finetune")
        print(f"\n  ── Epoch {epoch}/{cfg.train.max_epochs}  Phase={phase} ──────────")

        # Phase 4 LR reduction
        if epoch == cfg.train.phase3_end + 1:
            for pg in optimizer.param_groups: pg["lr"] *= 0.1
            print(f"  [Phase 4] LR → {optimizer.param_groups[0]['lr']:.1e}")

        # Update difficulty filter
        diff_max = get_difficulty_max(epoch, cfg)
        if diff_max != train_ds.max_difficulty:
            train_ds.max_difficulty = diff_max
            # Rebuild với augmentation flags giữ nguyên
            train_ds.sequences = [s for s in train_ds._all_sequences
                                   if diff_max is None or s["difficulty"] <= diff_max]
            print(f"  [Curriculum] max_diff={diff_max} → {len(train_ds)} seqs")

        train_loader = make_dataloader(train_ds, epoch, cfg, shuffle=True)

        # Train
        ts = train_epoch(model, train_loader, optimizer, criterion,
                         epoch, cfg, adap=adap, scaler=scaler)

        # Quick val loss
        model.eval(); vl_sum = 0.
        with torch.no_grad():
            for vb in val_loader:
                vb  = move_batch(vb, device)
                vo  = model(vb)
                vlo = criterion(vo, vb, epoch)
                vl_sum += vlo["loss"].item()
        avg_vl = vl_sum / max(len(val_loader), 1)

        scheduler.step()

        # BUG-3 FIX: print đúng cách (không dùng ternary expression làm giá trị)
        thr_str = f"  thr={adap.get():.3f}" if adap else ""
        sw_stats = criterion.step_weights.stats()
        lw_stats = criterion.loss_weights.stats()
        print(f"  train={ts['loss']:.4f}  val={avg_vl:.4f}{thr_str}"
              f"  sw_ratio={sw_stats['sw_ratio']:.2f}"
              f"  lw_pos={lw_stats['lw_pos']:.2f}"
              f"  lw_spd={lw_stats['lw_speed']:.2f}")

        # BUG-5 FIX: FAST VAL — chạy mỗi epoch, in metrics đầy đủ
        # (Subset nhỏ để nhanh: 500 samples hoặc toàn bộ nếu val nhỏ)
        fast_n = min(500, len(val_ds))
        fast_idx = random.sample(range(len(val_ds)), fast_n)
        from torch.utils.data import Subset
        fast_ds     = Subset(val_ds, fast_idx)
        fast_loader = DataLoader(fast_ds, batch_size=cfg.train.batch_size,
                                 shuffle=False, num_workers=0, drop_last=False)
        fast_metrics = evaluate(model, fast_loader, device,
                                tag=f"FAST-VAL ep{epoch} (n={fast_n})")

        # BUG-6 FIX: FULL VAL mỗi 3 epochs — in metrics đầy đủ
        if epoch % 3 == 0 or epoch == cfg.train.max_epochs:
            full_metrics = evaluate(model, val_loader, device,
                                    tag=f"FULL-VAL ep{epoch}")
            saver.update(full_metrics, epoch, model, optimizer, scheduler,
                         cfg, ts["loss"], avg_vl)
            history.append({"epoch": epoch, "train_loss": ts["loss"],
                             "val_loss": avg_vl, **full_metrics})
        else:
            # Save if fast val shows improvement
            saver.update(fast_metrics, epoch, model, optimizer, scheduler,
                         cfg, ts["loss"], avg_vl)

        # Periodic checkpoint
        if epoch % cfg.train.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.train.save_dir, f"ckpt_ep{epoch:03d}.pth"),
                epoch, model, optimizer, scheduler, fast_metrics, cfg)

        if saver.stop:
            print(f"  Early stop at epoch {epoch}."); break

    # Summary
    print(f"\n{'='*72}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best ADE: {saver.best_ade:.1f} km  (ST-Trans: {ST_TRANS['ADE']})")
    print(f"  Best 72h: {saver.best_72h:.1f} km  (ST-Trans: {ST_TRANS['72h']})")
    print(f"  Best ATE: {saver.best_ate:.1f} km  (ST-Trans: {ST_TRANS['ATE']})")
    print(f"  Best CTE: {saver.best_cte:.1f} km  (ST-Trans: {ST_TRANS['CTE']})")
    hist_path = os.path.join(cfg.train.save_dir, "history.json")
    with open(hist_path, "w") as f: json.dump(history, f, indent=2)
    print(f"  History: {hist_path}")
    print(f"{'='*72}\n")
    return saver.best_ade


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SRC-Track v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root",   default="TCND_vn")
    p.add_argument("--save_dir",    default="checkpoints/v2/")
    p.add_argument("--max_epochs",  type=int,   default=70)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--use_amp",     action="store_true")
    p.add_argument("--resume",      default=None)
    p.add_argument("--no_adaptive_thresh", action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    # Kaggle-style aliases
    p.add_argument("--dataset_root",  default=None)
    p.add_argument("--output_dir",    default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--num_epochs",    type=int,   default=None)
    p.add_argument("--regime_csv",    default=None)
    p.add_argument("--use_ot",        action="store_true")  # ignored, compat
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args = _apply_aliases(args)   # BUG-4 FIX: now defined above
    cfg  = get_config()

    cfg.data.data1d_train   = os.path.join(args.data_root, "Data1d/train")
    cfg.data.data1d_val     = os.path.join(args.data_root, "Data1d/val")
    cfg.data.data1d_test    = os.path.join(args.data_root, "Data1d/test")
    cfg.data.data3d_dir     = os.path.join(args.data_root, "Data3d")
    cfg.data.env_data_dir   = os.path.join(args.data_root, "Env_Data")
    cfg.train.save_dir      = args.save_dir
    cfg.train.max_epochs    = args.max_epochs
    cfg.train.batch_size    = args.batch_size
    cfg.train.lr            = args.lr
    cfg.train.num_workers   = args.num_workers
    cfg.train.seed          = args.seed
    if args.no_adaptive_thresh:
        cfg.train.use_adaptive_thresh = False
    if getattr(args, "regime_csv", None):
        cfg.data.regime_label_csv = args.regime_csv

    train(cfg, args)

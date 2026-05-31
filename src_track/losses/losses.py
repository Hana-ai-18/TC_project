"""
SRC-Track v2 — Loss Functions (v3: learned weights + full easy/hard split)

L_total = 0.5*L_easy + 0.5*L_hard
  L_easy = w_pos*L_pos(easy) + w_speed*L_speed(easy)
  L_hard = w_pos*L_pos(hard) + w_speed*L_speed(hard) + w_ep*L_endpoint(hard)
  + w_regime*L_regime  (from epoch 16, auxiliary)
  + w_div*L_diversity  (from epoch 31, auxiliary)

Weights w_pos, w_speed, w_ep learned via ConstrainedLossWeights.
Step weights learned via ConstrainedStepWeights (monotone ramp).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────
# Learned step weights — monotone increasing, ratio ≥ 3×
# ─────────────────────────────────────────────────────────────

class ConstrainedStepWeights(nn.Module):
    """
    Step weights w[0..11] that are:
    - Monotone: w[i+1] >= w[i]  (later timesteps more important)
    - Ratio w[-1]/w[0] >= ratio_min (72h >> 6h)
    - Mean = 1.0 (normalized)
    """
    def __init__(self, pred_len: int = 12, ratio_min: float = 3.0):
        super().__init__()
        self.pred_len  = pred_len
        self.ratio_min = ratio_min
        # Init: linear ramp from 0.5 to 2.0
        # Init: nearly uniform weights (ratio≈1.0)
        # Model learns 72h emphasis naturally from data
        # ratio_min=3.0 penalty will push ratio up gradually
        self.raw = nn.Parameter(torch.ones(pred_len) * 0.02)

    def forward(self) -> torch.Tensor:
        # cumsum(softplus) → strictly increasing
        w = torch.cumsum(F.softplus(self.raw), dim=0)
        # Normalize mean=1
        w = w * self.pred_len / (w.sum() + 1e-8)
        return w  # [12]

    def penalty(self, epoch: int = 100) -> torch.Tensor:
        """Soft penalty if ratio < ratio_min. Disabled for first 15 epochs."""
        if epoch < 15:
            return torch.zeros(1, device=self.raw.device).squeeze()
        w = self.forward()
        ratio = w[-1] / (w[0].clamp(min=1e-6))
        return 0.1 * F.relu(self.ratio_min - ratio) ** 2

    def stats(self) -> dict:
        with torch.no_grad():
            w = self.forward()
            return {
                "sw_6h":   w[0].item(),
                "sw_24h":  w[3].item() if len(w) > 3 else 0.,
                "sw_48h":  w[7].item() if len(w) > 7 else 0.,
                "sw_72h":  w[-1].item(),
                "sw_ratio": (w[-1] / w[0].clamp(1e-6)).item(),
            }


# ─────────────────────────────────────────────────────────────
# Learned loss weights — bounded by physics
# ─────────────────────────────────────────────────────────────

class ConstrainedLossWeights(nn.Module):
    """
    Loss term weights learned via softplus with bounds:
      w_pos   ∈ [0.5, 3.0]   position loss weight
      w_speed ∈ [0.1, 2.0]   speed loss weight  
      w_ep    ∈ [0.1, 2.0]   endpoint emphasis (hard storms only)
    """
    @staticmethod
    def _sp_inv(y: float) -> float:
        if y > 20.: return y
        return math.log(math.expm1(max(y, 1e-6)))

    def __init__(self, init_pos=1.0, init_speed=0.5, init_ep=0.5, anchor_w=0.02):
        super().__init__()
        self.anchor_w = anchor_w
        raw = torch.tensor([
            self._sp_inv(init_pos),
            self._sp_inv(init_speed),
            self._sp_inv(init_ep),
        ], dtype=torch.float)
        self.log_w = nn.Parameter(raw)
        self.register_buffer('log_w0', raw.clone())

    def w_pos(self)   -> torch.Tensor:
        return F.softplus(self.log_w[0]).clamp(0.5, 3.0)
    def w_speed(self) -> torch.Tensor:
        return F.softplus(self.log_w[1]).clamp(0.1, 5.0)  # FIX: was 2.0
    def w_ep(self)    -> torch.Tensor:
        return F.softplus(self.log_w[2]).clamp(0.1, 2.0)

    def penalty(self) -> torch.Tensor:
        return self.anchor_w * ((self.log_w - self.log_w0) ** 2).mean()

    def stats(self) -> dict:
        with torch.no_grad():
            return {
                "lw_pos":   self.w_pos().item(),
                "lw_speed": self.w_speed().item(),
                "lw_ep":    self.w_ep().item(),
            }


# ─────────────────────────────────────────────────────────────
# Primitive helpers
# ─────────────────────────────────────────────────────────────

def haversine_distance(pred: torch.Tensor, gt: torch.Tensor,
                       eps: float = 1e-8) -> torch.Tensor:
    """[B,T,2] lat°/lon° → [B,T] km"""
    R = 6371.0
    pr = torch.deg2rad(pred.float()); gr = torch.deg2rad(gt.float())
    dlat = pr[...,0] - gr[...,0]; dlon = pr[...,1] - gr[...,1]
    a = (torch.sin(dlat/2)**2
         + torch.cos(pr[...,0]) * torch.cos(gr[...,0]) * torch.sin(dlon/2)**2)
    return R * 2 * torch.asin(a.clamp(0., 1.).sqrt())  # FIX: clamp(0,1) not (eps,1-eps)


def compute_gt_speed(gt_traj: torch.Tensor) -> torch.Tensor:
    """[B,T,2] → [B,T-1] km/6h"""
    gt = gt_traj.float()
    dlat = gt[:,1:,0] - gt[:,:-1,0]
    dlon = gt[:,1:,1] - gt[:,:-1,1]
    lat_mid = (gt[:,1:,0] + gt[:,:-1,0]) / 2.0
    cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-3)
    return torch.sqrt((dlat*111.0)**2 + (dlon*111.0*cos_lat)**2)


# ─────────────────────────────────────────────────────────────
# Loss components
# ─────────────────────────────────────────────────────────────

def _position_loss_per_sample(pred_traj, gt_traj, step_weights,
                               regime_labels, rii_values,
                               huber_delta=300.0,  # FIX: was 100, must match ADE scale ~300km
                               rii_threshold=0.5, rii_weight_scale=1.5,
                               regime_b_extra=0.5) -> torch.Tensor:
    """Per-sample L_pos → [B]"""
    dist_km = haversine_distance(pred_traj, gt_traj)      # [B,T]
    w = step_weights.unsqueeze(0)                          # [1,T]
    diff_w = (1.0
              + rii_weight_scale * F.relu(rii_values.float() - rii_threshold)
              + regime_b_extra   * (regime_labels == 1).float())  # [B]
    weighted = dist_km * w
    huber = F.huber_loss(weighted, torch.zeros_like(weighted),
                         delta=huber_delta, reduction='none')     # [B,T]
    return (huber * diff_w.unsqueeze(1)).mean(dim=1) / huber_delta  # [B]


def _speed_loss_per_sample(pred_speed, gt_traj) -> torch.Tensor:
    """Per-sample log-space speed loss → [B]"""
    gt_speed = compute_gt_speed(gt_traj)
    T = min(pred_speed.shape[1], gt_speed.shape[1])
    ps = pred_speed[:, :T].float()
    gs = gt_speed[:,  :T].float()
    return F.mse_loss(torch.log(ps+1.), torch.log(gs+1.),
                      reduction='none').mean(dim=1)  # [B]


def _endpoint_loss(pred_traj, gt_traj, d=300.) -> torch.Tensor:
    """48h (step 7) + 72h (step 11) emphasis → scalar"""
    T = pred_traj.shape[1]
    total = pred_traj.new_zeros(())
    w_sum = 0.
    for s, w in [(7, 0.5), (11, 1.5)]:
        if s < T:
            d_s = haversine_distance(pred_traj[:,s:s+1], gt_traj[:,s:s+1]).squeeze(1)
            total = total + w * torch.where(
                d_s < d, d_s**2/(2*d), d_s - d/2.).mean() / d
            w_sum += w
    return total / max(w_sum, 1e-6)


def regime_loss(regime_logits, regime_labels,
                label_smoothing=0.05) -> torch.Tensor:
    return F.cross_entropy(regime_logits.float(), regime_labels,
                           label_smoothing=label_smoothing)


def diversity_loss(expert_dirs, regime_labels,
                   threshold=-0.5) -> torch.Tensor:
    dir_A = expert_dirs['A'].float()
    dir_C = expert_dirs['C'].float()
    cos_sim  = (dir_A * dir_C).sum(dim=-1)              # [B,T]
    mask_b   = (regime_labels == 1).float().unsqueeze(1)
    return (F.relu(cos_sim - threshold) * mask_b).mean()


# ─────────────────────────────────────────────────────────────
# Main loss class
# ─────────────────────────────────────────────────────────────

class SRCTrackLoss(nn.Module):
    """
    Full SRC-Track loss with:
    - Learned step weights (ConstrainedStepWeights)
    - Learned loss weights (ConstrainedLossWeights)  
    - Easy/Hard 50/50 split on ALL main loss terms
    - Curriculum: L_regime from epoch 16, L_diversity from epoch 31
    - Fixed auxiliary weights: w_regime=0.1, w_div=0.05

    easy_thresh: mutable, updated by AdaptiveThreshold in trainer
    """

    def __init__(self,
                 # Speed weight — critical for ATE fix
                 w_speed:  float = 5.0,    # was 0.5 → 10x amplify
                 # Auxiliary (fixed)
                 w_regime: float = 0.5,    # was 0.1 → RC needs more signal
                 w_div:    float = 0.05,
                 regime_start_epoch: int = 16,
                 div_start_epoch:    int = 31,
                 # L_pos params
                 huber_delta:      float = 300.0,   # FIX: was 100
                 rii_threshold:    float = 0.5,
                 rii_weight_scale: float = 1.5,
                 regime_b_extra:   float = 0.5,
                 # Diversity
                 diversity_threshold: float = -0.5,
                 label_smoothing:     float = 0.05,
                 # Easy/Hard threshold (mutable, updated by AdaptiveThreshold)
                 easy_thresh: float = 0.40,
                 # Pred length
                 pred_len: int = 12,
                 # unused kwargs for compat
                 step_w_min: float = 0.625,
                 step_w_max: float = 2.0,
                 **kwargs):
        super().__init__()

        # Learned weights — registered as submodules so optimizer sees them
        self.step_weights = ConstrainedStepWeights(pred_len)
        self.loss_weights = ConstrainedLossWeights()

        # Weights
        self.w_speed  = w_speed
        # Fixed auxiliary
        self.w_regime = w_regime
        self.w_div    = w_div
        self.regime_start = regime_start_epoch
        self.div_start    = div_start_epoch

        # L_pos kwargs
        self.pos_kw = dict(
            huber_delta=huber_delta,
            rii_threshold=rii_threshold,
            rii_weight_scale=rii_weight_scale,
            regime_b_extra=regime_b_extra,
        )
        self.diversity_threshold = diversity_threshold
        self.label_smoothing     = label_smoothing
        self.easy_thresh = easy_thresh  # updated by AdaptiveThreshold

    def forward(self, outputs: Dict, batch: Dict,
                current_epoch: int) -> Dict:

        pred_traj     = outputs['pred_traj']
        pred_speed    = outputs['pred_speed']
        regime_logits = outputs['regime_logits']
        expert_dirs   = outputs['expert_dirs']

        gt_traj       = batch['gt_traj']
        regime_labels = batch['regime_label']
        rii_values    = batch['rii']
        difficulty    = batch.get('difficulty')

        # Learned weights
        sw      = self.step_weights()    # [T]
        w_pos   = self.loss_weights.w_pos()
        # FIX: multiply learned weight by fixed amplifier self.w_speed
        w_speed = self.loss_weights.w_speed() * self.w_speed
        w_ep    = self.loss_weights.w_ep()

        # Per-sample losses [B]
        pos_per   = _position_loss_per_sample(
            pred_traj, gt_traj, sw, regime_labels, rii_values, **self.pos_kw)
        speed_per = _speed_loss_per_sample(pred_speed, gt_traj)

        # Easy/Hard split
        easy_mask = None; easy_frac = 1.0
        if difficulty is not None:
            easy_mask = (difficulty < self.easy_thresh)
            easy_frac = float(easy_mask.float().mean().item())
        hard_mask = ~easy_mask if easy_mask is not None else None

        def _mean(tensor, mask):
            if mask is None or not mask.any():
                return tensor.mean()
            return tensor[mask].mean() if mask.any() else tensor.mean()

        # Easy loss (position + speed only)
        l_pos_easy   = _mean(pos_per,   easy_mask)
        l_speed_easy = _mean(speed_per, easy_mask)
        L_easy = w_pos * l_pos_easy + w_speed * l_speed_easy

        # Hard loss (position + speed + endpoint emphasis)
        l_pos_hard   = _mean(pos_per,   hard_mask)
        l_speed_hard = _mean(speed_per, hard_mask)

        if hard_mask is not None and hard_mask.any():
            l_ep = _endpoint_loss(
                pred_traj[hard_mask], gt_traj[hard_mask])
        else:
            l_ep = pred_traj.new_zeros(())

        L_hard = w_pos * l_pos_hard + w_speed * l_speed_hard + w_ep * l_ep

        # 50/50 blend
        L_main = 0.5 * L_easy + 0.5 * L_hard

        # Auxiliary losses (curriculum)
        if current_epoch >= self.regime_start:
            l_regime = regime_loss(regime_logits, regime_labels,
                                   self.label_smoothing)
        else:
            l_regime = pred_traj.new_zeros(())

        if current_epoch >= self.div_start:
            l_div = diversity_loss(expert_dirs, regime_labels,
                                   self.diversity_threshold)
        else:
            l_div = pred_traj.new_zeros(())

        # Regularization on learned weights
        l_reg = self.step_weights.penalty(current_epoch) + self.loss_weights.penalty()

        total = (L_main
                 + self.w_regime * l_regime
                 + self.w_div    * l_div
                 + l_reg)

        def _s(x):
            return x.item() if torch.is_tensor(x) else float(x)

        sw_s = self.step_weights.stats()
        lw_s = self.loss_weights.stats()

        return {
            "loss":       total,
            "l_pos":      _s(0.5*(l_pos_easy + l_pos_hard)),
            "l_speed":    _s(0.5*(l_speed_easy + l_speed_hard)),
            "l_regime":   _s(l_regime),
            "l_div":      _s(l_div),
            "l_ep":       _s(l_ep),
            "L_easy":     _s(L_easy),
            "L_hard":     _s(L_hard),
            "easy_frac":  easy_frac,
            # [BUG-D FIX] stats() already has 'sw_*' and 'lw_*' prefixes
            # f'sw_{k}' was producing 'sw_sw_72h' etc. Now using keys directly.
            **sw_s,   # keys: sw_6h, sw_24h, sw_48h, sw_72h, sw_ratio
            **lw_s,   # keys: lw_pos, lw_speed, lw_ep
        }

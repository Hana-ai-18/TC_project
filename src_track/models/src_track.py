"""
SRC-Track — Full Model Architecture
Modules:
  SpatialContextEncoder (SCE)     : ViT-lite + annular attention
  TemporalKinematicEncoder (TKE)  : Transformer + regime tokens
  RegimeClassifier (RC)           : 3-class softmax
  SpeedHead                       : log-space speed prediction
  TrajectoryExpert                : single expert decoder
  RegimeConditionedDecoder (MoE)  : 3-expert soft blend
  SRCTrack                        : full model
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────
# 0.  Helpers
# ─────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split [B, C, H, W] into non-overlapping patches → [B, n_patches, C*patch²]"""
    def __init__(self, in_ch: int, d_model: int, patch_size: int, img_size: int):
        super().__init__()
        self.patch_size = patch_size
        n_patches_1d = img_size // patch_size
        self.n_patches = n_patches_1d ** 2
        self.proj = nn.Conv2d(in_ch, d_model,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, d_model, n_h, n_w] → [B, n_patches, d_model]
        x = self.proj(x)
        B, D, nh, nw = x.shape
        return x.flatten(2).transpose(1, 2)   # [B, n_patches, d_model]


class SimpleMHA(nn.Module):
    """Thin wrapper around nn.MultiheadAttention for [B, N, D] (batch_first)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        h, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm(x + h)
        # FFN
        x = self.norm2(x + self.ff(x))
        return x


# ─────────────────────────────────────────────────────────────
# 1.  SPATIAL CONTEXT ENCODER (SCE)
#     Input : [B, T_obs, 13, 81, 81]
#     Output: [B, 128]   (annular-aware spatial context)
# ─────────────────────────────────────────────────────────────

class SpatialContextEncoder(nn.Module):
    """
    ViT-lite with:
      - Steering branch (4 channels: GPH500, U500, V500, V850)
        → patch embed → Transformer with annular attention bias
      - Thermo branch (9 channels: rest)
        → Conv + Global Pool → [B, 32]
      - Cross-attention: steering queries thermo context
    Output: [B, d_model=128]
    """

    STEERING_CH = [1, 5, 9, 10]  # GPH500,U500,V500,V850 (TCND order)
    THERMO_CH   = [0, 2, 3, 4, 6, 7, 8, 11, 12]

    def __init__(
        self,
        d_model:    int = 128,
        n_heads:    int = 4,
        n_layers:   int = 4,
        patch_size: int = 8,
        img_size:   int = 81,
        thermo_dim: int = 32,
        dropout:    float = 0.1,
        inner_deg:  float = 3.0,
        outer_deg:  float = 7.0,
        cell_deg:   float = 0.25,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers

        n_steer = len(self.STEERING_CH)
        n_therm = len(self.THERMO_CH)

        # ── Steering branch ───────────────────────────────────
        self.steer_embed = PatchEmbed(n_steer, d_model, patch_size, img_size)
        n_patches = self.steer_embed.n_patches

        # Positional embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer layers
        self.steer_layers = nn.ModuleList([
            SimpleMHA(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.steer_pool = nn.Linear(n_patches, 1)   # learned pooling

        # ── Thermo branch ─────────────────────────────────────
        self.thermo_proj = nn.Sequential(
            nn.Conv2d(n_therm, thermo_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),   # [B, thermo_dim, 1, 1]
        )
        self.thermo_expand = nn.Linear(thermo_dim, d_model)

        # ── Cross-attention: steering → thermo ────────────────
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        # ── Temporal pooling (over T_obs) ─────────────────────
        self.temporal_pool = nn.Linear(d_model, d_model)

        # ── Pre-compute annular attention bias ────────────────
        annular_mask = self._compute_annular_mask(
            img_size, patch_size, inner_deg, outer_deg, cell_deg
        )  # [n_patches] bool
        # bias: annular patches → 0, non-annular → -1e4
        attn_bias = torch.zeros(n_patches)
        attn_bias[~annular_mask] = -1e4
        # Expand to [n_patches, n_patches] additive bias for self-attention
        # We use a simpler key-masking: non-annular patches are masked as keys
        self.register_buffer('annular_key_mask', ~annular_mask)  # True = ignore

    @staticmethod
    def _compute_annular_mask(
        img_size:   int,
        patch_size: int,
        inner_deg:  float,
        outer_deg:  float,
        cell_deg:   float,
    ) -> torch.Tensor:
        inner_cells = inner_deg / cell_deg
        outer_cells = outer_deg / cell_deg
        center = img_size // 2
        n1d = img_size // patch_size
        mask = torch.zeros(n1d, n1d, dtype=torch.bool)
        for i in range(n1d):
            for j in range(n1d):
                ci = (i + 0.5) * patch_size
                cj = (j + 0.5) * patch_size
                dist = math.sqrt((ci - center)**2 + (cj - center)**2)
                if inner_cells <= dist <= outer_cells:
                    mask[i, j] = True
        return mask.flatten()   # [n_patches]

    def _encode_one(self, data3d_t: torch.Tensor) -> torch.Tensor:
        """
        data3d_t: [B, 13, H, W]  (single timestep)
        Returns:  [B, d_model]
        """
        B = data3d_t.shape[0]

        # ── Steering branch ───────────────────────────────────
        steer_in = data3d_t[:, self.STEERING_CH]           # [B, 4, H, W]
        patches  = self.steer_embed(steer_in) + self.pos_embed  # [B, N, D]

        for layer in self.steer_layers:
            # key_padding_mask: True = ignore  [B, N] — non-annular patches
            key_mask = self.annular_key_mask.unsqueeze(0).expand(B, -1)
            patches  = layer(patches, attn_mask=None)
            # Soft masking: zero out non-annular patch outputs
            # (hard masking is done via key_padding_mask in layer.attn)

        # Pooling: [B, N, D] → [B, D]
        steer_out = self.steer_pool(patches.transpose(1, 2)).squeeze(-1)  # [B, D]

        # ── Thermo branch ─────────────────────────────────────
        thermo_in  = data3d_t[:, self.THERMO_CH]           # [B, 9, H, W]
        thermo_out = self.thermo_proj(thermo_in).squeeze(-1).squeeze(-1)  # [B, td]
        thermo_exp = self.thermo_expand(thermo_out)         # [B, D]

        # ── Cross-attention ───────────────────────────────────
        q   = steer_out.unsqueeze(1)                        # [B, 1, D]
        kv  = thermo_exp.unsqueeze(1)                       # [B, 1, D]
        out, _ = self.cross_attn(q, kv, kv)                # [B, 1, D]
        out = self.cross_norm(steer_out + out.squeeze(1))   # [B, D]

        return out   # [B, D]

    def forward(self, data3d: torch.Tensor) -> torch.Tensor:
        """
        data3d: [B, T_obs, 13, H, W]
        Returns: [B, d_model]
        """
        B, T = data3d.shape[:2]
        encoded = torch.stack(
            [self._encode_one(data3d[:, t]) for t in range(T)], dim=1
        )  # [B, T, D]
        # Mean pool over time
        ctx = encoded.mean(dim=1)           # [B, D]
        return self.temporal_pool(ctx)      # [B, D]


# ─────────────────────────────────────────────────────────────
# 2.  TEMPORAL KINEMATIC ENCODER (TKE)
#     Input : physnorm [B,T,9] + sve [B,T,24] + env_data [B,T,84]
#     Output: kinematic_ctx [B,64], regime_ctx [B,64]
# ─────────────────────────────────────────────────────────────

class TemporalKinematicEncoder(nn.Module):
    """
    Transformer with 3 learnable Physical Regime Tokens prepended.
    Sequence: [tok_A | tok_B | tok_C | obs_0 | ... | obs_{T-1}]
    Output: regime_ctx from token positions, kinematic_ctx from last obs.
    """

    def __init__(
        self,
        input_dim: int = 117,   # 9 + 24 + 84
        d_model:   int = 64,
        n_heads:   int = 4,
        n_layers:  int = 4,
        dropout:   float = 0.1,
        obs_len:   int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_regime_tokens = 3
        self.seq_len = self.n_regime_tokens + obs_len   # 11

        self.input_proj   = nn.Linear(input_dim, d_model)
        self.regime_tokens = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.pos_embedding = nn.Embedding(self.seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        # Pool the 3 regime tokens → single regime_ctx
        self.regime_pool = nn.Linear(3 * d_model, d_model)

    def forward(
        self,
        physnorm:  torch.Tensor,   # [B, T, 9]
        sve:       torch.Tensor,   # [B, T, 24]
        env_data:  torch.Tensor,   # [B, T, 84]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = physnorm.shape

        # Concat inputs → [B, T, 117]
        x = torch.cat([physnorm, sve, env_data], dim=-1)
        x = self.input_proj(x)   # [B, T, d_model]

        # Prepend regime tokens
        tok = self.regime_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 3, D]
        x   = torch.cat([tok, x], dim=1)                          # [B, 11, D]

        # Positional encoding
        pos = torch.arange(self.seq_len, device=x.device)
        x   = x + self.pos_embedding(pos)                         # [B, 11, D]

        # Transformer
        out = self.transformer(x)                                  # [B, 11, D]

        # Extract outputs
        regime_raw   = out[:, :3, :].reshape(B, 3 * self.d_model) # [B, 3D]
        regime_ctx   = self.regime_pool(regime_raw)                # [B, D]
        kinematic_ctx = out[:, -1, :]                              # [B, D]

        return kinematic_ctx, regime_ctx


# ─────────────────────────────────────────────────────────────
# 3.  REGIME CLASSIFIER (RC)
# ─────────────────────────────────────────────────────────────

class RegimeClassifier(nn.Module):
    """
    3-class classifier: Regime A (subtropical high) / B (transition) / C (westerlies).
    Input : context [B, 256]
    Output: regime_probs [B, 3], regime_logits [B, 3]
    """

    def __init__(self, d_context: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_context, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(context)                    # [B, 3]
        probs  = F.softmax(logits, dim=-1)            # [B, 3]
        return probs, logits


# ─────────────────────────────────────────────────────────────
# 4.  SPEED HEAD
# ─────────────────────────────────────────────────────────────

class SpeedHead(nn.Module):
    """
    Predict translation speed for each future timestep.
    Uses log-space output to balance slow/fast storms.
    Input : context [B, 256]
    Output: pred_speed [B, T_pred]  in km/6h, range (speed_min, speed_max)
    """

    def __init__(
        self,
        d_context: int = 256,
        pred_len:  int = 12,
        speed_min: float = 3.0,
        speed_max: float = 100.0,
        hidden:    int = 128,
    ):
        super().__init__()
        self.speed_min = speed_min
        self.speed_max = speed_max

        self.net = nn.Sequential(
            nn.Linear(d_context, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, pred_len),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.net(context)   # [B, T_pred]
        # Softplus keeps output positive; scale to physical range
        # F.softplus(x) ∈ (0, ∞); * 5 + speed_min → (speed_min, ~large)
        speed = F.softplus(raw) * 5.0 + self.speed_min
        speed = torch.clamp(speed, self.speed_min, self.speed_max)
        return speed   # [B, T_pred]  km/6h


# ─────────────────────────────────────────────────────────────
# 5.  TRAJECTORY EXPERT  (one expert in the MoE)
# ─────────────────────────────────────────────────────────────

class TrajectoryExpert(nn.Module):
    """
    Predicts a unit direction vector for a single decode step.
    Uses a 2-layer TransformerDecoder: query = current state,
    memory = global context.

    Input:
      state   [B, 5]  = (lat_rel, lon_rel, speed_prev_n, heading_sin, heading_cos)
      context [B, D_ctx]
    Output: direction unit vector [B, 2]  (lat_component, lon_component)
    """

    def __init__(self, d_context: int = 256, d_model: int = 64,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.state_proj   = nn.Linear(5, d_model)
        self.context_proj = nn.Linear(d_context, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.decoder    = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q   = self.state_proj(state).unsqueeze(1)     # [B, 1, D]
        mem = self.context_proj(context).unsqueeze(1) # [B, 1, D]
        out = self.decoder(q, mem).squeeze(1)          # [B, D]
        raw = self.output_proj(out)                    # [B, 2]
        return F.normalize(raw, dim=-1)                # unit vector [B, 2]


# ─────────────────────────────────────────────────────────────
# 6.  REGIME-CONDITIONED MOE DECODER (RCPD)
# ─────────────────────────────────────────────────────────────

class RegimeConditionedDecoder(nn.Module):
    """
    3-Expert Mixture-of-Experts autoregressive decoder.

    At each step t:
      1. Each expert predicts a direction unit vector.
      2. Blend = P(A)*dir_A + P(B)*dir_B + P(C)*dir_C
      3. Normalise blend → unit vector.
      4. Apply speed from SpeedHead to get displacement.
      5. Update position.

    Returns:
      pred_traj  [B, T_pred, 2]  in physical (lat°, lon°)
      expert_dirs Dict[str, Tensor[B, T_pred, 2]]  for diversity loss
    """

    def __init__(
        self,
        d_context: int = 256,
        d_model:   int = 64,
        n_layers:  int = 2,
        pred_len:  int = 12,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.expert_A = TrajectoryExpert(d_context, d_model, n_layers, dropout)
        self.expert_B = TrajectoryExpert(d_context, d_model, n_layers, dropout)
        self.expert_C = TrajectoryExpert(d_context, d_model, n_layers, dropout)

    @staticmethod
    def _encode_state(
        pos:          torch.Tensor,   # [B, 2] (lat°, lon°)
        speed_prev:   torch.Tensor,   # [B]   km/6h
        heading_prev: torch.Tensor,   # [B]   radians
        origin:       torch.Tensor,   # [B, 2] reference position
    ) -> torch.Tensor:
        """Build 5-dim state vector for expert query."""
        lat_rel = (pos[:, 0] - origin[:, 0]) / 10.0     # normalise by ~10°
        lon_rel = (pos[:, 1] - origin[:, 1]) / 10.0
        spd_n   = (speed_prev - 18.0) / 8.0             # SCS-normalised
        h_sin   = torch.sin(heading_prev)
        h_cos   = torch.cos(heading_prev)
        return torch.stack([lat_rel, lon_rel, spd_n, h_sin, h_cos], dim=-1)  # [B, 5]

    def forward(
        self,
        context:       torch.Tensor,   # [B, 256]
        regime_probs:  torch.Tensor,   # [B, 3]   P(A), P(B), P(C)
        pred_speed:    torch.Tensor,   # [B, T_pred]  km/6h
        last_pos:      torch.Tensor,   # [B, 2]  physical (lat°, lon°)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B = context.shape[0]
        device = context.device

        pos      = last_pos.clone()                        # [B, 2]
        origin   = last_pos.clone()                        # reference for relative pos
        heading  = torch.zeros(B, device=device)           # initial heading = 0
        speed_p  = torch.full((B,), 18.0, device=device)  # initial prev speed

        trajectory = []
        dirs_A, dirs_B, dirs_C = [], [], []

        for t in range(self.pred_len):
            state = self._encode_state(pos, speed_p, heading, origin)  # [B, 5]

            dA = self.expert_A(state, context)   # [B, 2] unit vectors
            dB = self.expert_B(state, context)
            dC = self.expert_C(state, context)

            dirs_A.append(dA)
            dirs_B.append(dB)
            dirs_C.append(dC)

            # Soft blend
            pA = regime_probs[:, 0:1]   # [B, 1]
            pB = regime_probs[:, 1:2]
            pC = regime_probs[:, 2:3]
            dir_blend = pA * dA + pB * dB + pC * dC     # [B, 2]
            dir_blend = F.normalize(dir_blend, dim=-1)   # re-normalise

            # Speed at step t
            spd = pred_speed[:, t]   # [B]  km/6h, already clamped

            # Displacement in degrees
            lat_rad = torch.deg2rad(pos[:, 0])
            # dir_blend: [lat_component, lon_component]
            delta_lat = spd * dir_blend[:, 0] / 111.0
            delta_lon = spd * dir_blend[:, 1] / (111.0 * torch.cos(lat_rad).clamp(min=1e-3))

            pos = pos + torch.stack([delta_lat, delta_lon], dim=-1)

            # Update heading for next step
            heading  = torch.atan2(dir_blend[:, 1], dir_blend[:, 0])
            speed_p  = spd

            trajectory.append(pos.clone())

        pred_traj = torch.stack(trajectory, dim=1)   # [B, T_pred, 2]
        expert_dirs = {
            'A': torch.stack(dirs_A, dim=1),   # [B, T_pred, 2]
            'B': torch.stack(dirs_B, dim=1),
            'C': torch.stack(dirs_C, dim=1),
        }
        return pred_traj, expert_dirs


# ─────────────────────────────────────────────────────────────
# 7.  CONTEXT FUSION
# ─────────────────────────────────────────────────────────────

class ContextFusion(nn.Module):
    """Fuse SCE + TKE kinematic + TKE regime → context [B, 256]"""

    def __init__(self, sce_dim: int = 128, tke_dim: int = 64):
        super().__init__()
        in_dim = sce_dim + tke_dim + tke_dim   # 256
        self.proj = nn.Linear(in_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, sce_out: torch.Tensor,
                kinematic_ctx: torch.Tensor,
                regime_ctx:    torch.Tensor) -> torch.Tensor:
        cat = torch.cat([sce_out, kinematic_ctx, regime_ctx], dim=-1)
        return self.norm(self.proj(cat))


# ─────────────────────────────────────────────────────────────
# 8.  FULL SRC-TRACK MODEL
# ─────────────────────────────────────────────────────────────

class SRCTrack(nn.Module):
    """
    Full SRC-Track model.

    Forward returns:
      pred_traj    [B, T_pred, 2]   physical (lat°, lon°)
      pred_speed   [B, T_pred]      km/6h
      regime_probs [B, 3]
      regime_logits[B, 3]
      expert_dirs  Dict[str, Tensor[B, T_pred, 2]]
    """

    def __init__(
        self,
        # SCE
        sce_d_model:    int = 128,
        sce_n_heads:    int = 4,
        sce_n_layers:   int = 4,
        sce_patch_size: int = 8,
        sce_img_size:   int = 81,
        sce_thermo_dim: int = 32,
        # TKE
        tke_input_dim: int = 117,
        tke_d_model:   int = 64,
        tke_n_heads:   int = 4,
        tke_n_layers:  int = 4,
        obs_len:       int = 8,
        # Classifier / heads
        rc_dropout:   float = 0.3,
        speed_hidden: int   = 128,
        speed_min:    float = 3.0,
        speed_max:    float = 100.0,
        # Decoder
        expert_d_model:  int = 64,
        expert_n_layers: int = 2,
        pred_len:        int = 12,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len

        # Modules
        self.sce = SpatialContextEncoder(
            d_model=sce_d_model, n_heads=sce_n_heads, n_layers=sce_n_layers,
            patch_size=sce_patch_size, img_size=sce_img_size,
            thermo_dim=sce_thermo_dim, dropout=dropout,
        )
        self.tke = TemporalKinematicEncoder(
            input_dim=tke_input_dim, d_model=tke_d_model,
            n_heads=tke_n_heads, n_layers=tke_n_layers,
            dropout=dropout, obs_len=obs_len,
        )
        self.fusion = ContextFusion(sce_dim=sce_d_model, tke_dim=tke_d_model)

        context_dim = sce_d_model + tke_d_model + tke_d_model   # 256

        self.regime_cls = RegimeClassifier(d_context=context_dim, dropout=rc_dropout)
        self.speed_head = SpeedHead(d_context=context_dim, pred_len=pred_len,
                                    speed_min=speed_min, speed_max=speed_max,
                                    hidden=speed_hidden)
        self.decoder    = RegimeConditionedDecoder(
            d_context=context_dim, d_model=expert_d_model,
            n_layers=expert_n_layers, pred_len=pred_len, dropout=dropout,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch keys (all on same device):
          'physnorm'   [B, T_obs, 9]
          'data3d'     [B, T_obs, 13, H, W]
          'env_data'   [B, T_obs, 84]
          'sve'        [B, T_obs, 24]
          'last_pos'   [B, 2]   physical (lat°, lon°)
        """
        physnorm = batch['physnorm']
        data3d   = batch['data3d']
        env_data = batch['env_data']
        sve      = batch['sve']
        last_pos = batch['last_pos']

        # ── Encode ────────────────────────────────────────────
        sce_out                = self.sce(data3d)                     # [B, 128]
        kinematic_ctx, reg_ctx = self.tke(physnorm, sve, env_data)   # [B,64] each
        context                = self.fusion(sce_out, kinematic_ctx, reg_ctx)  # [B, 256]

        # ── Heads ─────────────────────────────────────────────
        regime_probs, regime_logits = self.regime_cls(context)   # [B,3], [B,3]
        pred_speed                  = self.speed_head(context)   # [B, T_pred]

        # ── Decode ────────────────────────────────────────────
        pred_traj, expert_dirs = self.decoder(
            context, regime_probs, pred_speed, last_pos
        )   # [B, T_pred, 2], dict

        return {
            'pred_traj':     pred_traj,        # [B, T_pred, 2]  lat°, lon°
            'pred_speed':    pred_speed,        # [B, T_pred]  km/6h
            'regime_probs':  regime_probs,      # [B, 3]
            'regime_logits': regime_logits,     # [B, 3]
            'expert_dirs':   expert_dirs,       # {'A','B','C'}: [B, T_pred, 2]
            'context':       context,           # [B, 256] for Jacobian
        }

    @torch.no_grad()
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# 9.  PHYSICS SENSITIVITY ELLIPSE  (inference, no training)
# ─────────────────────────────────────────────────────────────

def compute_sensitivity_ellipse(
    model:         SRCTrack,
    batch:         Dict[str, torch.Tensor],
    sigma_steering: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Compute physics-based uncertainty ellipse at 72h endpoint.
    Uses Jacobian ∂(endpoint_72h) / ∂(steering_vector).

    sigma_steering: [2, 2] covariance of steering flow.
                    If None, uses identity (unit perturbation).

    Returns dict with:
      major_axis_km, minor_axis_km, orientation [2,2], ellipse_cov [2,2]
    """
    model.eval()

    # We need grad w.r.t. the SVE steering components (u_steer, v_steer)
    # Represented as batch['sve'][:, -1, 0:2]  (last obs step, dims 0-1)
    sve_orig = batch['sve'].clone()

    # Use only the first sample for Jacobian (single-storm inference)
    single_batch = {k: v[:1] for k, v in batch.items()}

    # Make the steering inputs differentiable
    sv = single_batch['sve'][:, -1, :2].detach().requires_grad_(True)   # [1, 2]
    single_batch['sve'] = single_batch['sve'].clone()
    single_batch['sve'][:, -1, :2] = sv

    with torch.enable_grad():
        out       = model(single_batch)
        pred_traj = out['pred_traj']          # [1, T_pred, 2]
        endpoint  = pred_traj[0, -1]          # [2]  lat, lon at 72h

        J = torch.zeros(2, 2, device=sv.device)
        for i in range(2):
            grad = torch.autograd.grad(
                endpoint[i], sv,
                retain_graph=(i < 1),
                create_graph=False,
            )[0]   # [1, 2]
            J[i] = grad.squeeze(0)

    if sigma_steering is None:
        sigma_steering = torch.eye(2, device=J.device)

    # ellipse_cov = J @ Σ_steering @ J^T
    ellipse_cov = J @ sigma_steering.to(J.device) @ J.T   # [2, 2]
    # Symmetrize to avoid numerical issues
    ellipse_cov = (ellipse_cov + ellipse_cov.T) / 2.0
    # Clamp negative eigenvalues
    eigvals, eigvecs = torch.linalg.eigh(ellipse_cov)
    eigvals = torch.clamp(eigvals, min=0.0)

    major_axis_km = (eigvals[1].sqrt() * 111.0).item()
    minor_axis_km = (eigvals[0].sqrt() * 111.0).item()

    return {
        'major_axis_km': major_axis_km,
        'minor_axis_km': minor_axis_km,
        'orientation':   eigvecs.detach().cpu(),      # [2, 2]
        'ellipse_cov':   ellipse_cov.detach().cpu(),  # [2, 2]
        'jacobian':      J.detach().cpu(),
    }


# ─────────────────────────────────────────────────────────────
# 10.  MODEL FACTORY
# ─────────────────────────────────────────────────────────────

def build_model(cfg=None) -> SRCTrack:
    """Instantiate SRCTrack from config or with defaults."""
    if cfg is None:
        return SRCTrack()
    m = cfg.model
    return SRCTrack(
        sce_d_model=m.sce_d_model,   sce_n_heads=m.sce_n_heads,
        sce_n_layers=m.sce_n_layers, sce_patch_size=m.sce_patch_size,
        sce_thermo_dim=m.sce_thermo_dim,
        tke_input_dim=m.tke_input_dim, tke_d_model=m.tke_d_model,
        tke_n_heads=m.tke_n_heads,   tke_n_layers=m.tke_n_layers,
        obs_len=cfg.data.obs_len,
        rc_dropout=0.3,
        speed_hidden=m.speed_hidden, speed_min=m.speed_min, speed_max=m.speed_max,
        expert_d_model=m.expert_d_model, expert_n_layers=m.expert_n_layers,
        pred_len=cfg.data.pred_len,
        dropout=m.dropout,
    )

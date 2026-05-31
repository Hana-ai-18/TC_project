"""
SRC-Track v2 — Config
Steering Regime-Conditioned TC Track Forecasting for SCS

CHANGES vs v1:
  - VELOCITY_NORM fixed: 1219.84 → 100.0 (SCS max, was WNP max — root cause of ATE=213km)
  - Easy/Hard curriculum added (thầy's suggestion)
  - AdaptiveThreshold integrated
  - Channel mapping updated to match actual TCND data3d
"""
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    # Paths
    data1d_train:     str = "Data1d/train"
    data1d_val:       str = "Data1d/val"
    data1d_test:      str = "Data1d/test"
    data3d_dir:       str = "Data3d"
    env_data_dir:     str = "Env_Data"
    regime_label_csv: str = "sequence_regime_labels.csv"

    # Sequence windows
    obs_len:  int = 8
    pred_len: int = 12
    stride:   int = 1

    # FIX: SCS-specific kinematic stats
    # OLD: VELOCITY_NORM = 1219.84 (WNP — caused ATE=213km speed bias)
    # NEW: SCS max ≈ 80–100 km/6h
    scs_speed_mean:  float = 18.0
    scs_speed_std:   float = 8.0
    scs_velocity_norm: float = 100.0   # FIX: was 1219.84
    delta_speed_std: float = 5.0
    curvature_norm:  float = 0.7854    # π/4

    # ERA5 patch
    patch_cells: int   = 81
    n_channels:  int   = 13
    cell_deg:    float = 0.25

    # Channel mapping — TCND Data3d (from DATA3D_MEAN/STD in trajectoriesWithMe_unet_training)
    # 0:GPH_200  1:GPH_500  2:GPH_850  3:GPH_925
    # 4:U_200    5:U_500    6:U_850    7:U_925
    # 8:V_200    9:V_500   10:V_850   11:V_925
    # 12:SST
    steering_ch: List[int] = field(default_factory=lambda: [1, 5, 9, 10])  # GPH500, U500, V500, V850
    thermo_ch:   List[int] = field(default_factory=lambda: [0, 2, 3, 4, 6, 7, 8, 11, 12])

    # Annular steering ring
    annular_inner_deg: float = 3.0
    annular_outer_deg: float = 7.0

    # SVE
    sve_dim: int = 26   # +2: thickness proxy + upper divergence
    env_dim: int = 84

    # Augmentation
    use_stride_aug: bool = True
    aug_strides:    List[int] = field(default_factory=lambda: [2, 3])


@dataclass
class ModelConfig:
    # SCE
    sce_d_model:    int = 128
    sce_n_heads:    int = 4
    sce_n_layers:   int = 4
    sce_patch_size: int = 8
    sce_img_size:   int = 81
    sce_thermo_dim: int = 32
    sce_dropout:    float = 0.1

    # TKE: input = PhysNorm(9) + SVE(24) + Env_data(84) = 117
    tke_input_dim: int = 119   # 9(PhysNorm)+26(SVE)+84(Env) — SVE now 26
    tke_d_model:   int = 64
    tke_n_heads:   int = 4
    tke_n_layers:  int = 4

    # Context: SCE(128) + kinematic(64) + regime(64) = 256
    context_dim: int = 256

    # RC
    rc_dropout: float = 0.3

    # Speed Head
    speed_hidden: int   = 128
    speed_min:    float = 3.0
    speed_max:    float = 150.0   # FIX: was 100. gt_speed=113 → capped → bias can never close

    # MoE Decoder
    expert_d_model:  int = 64
    expert_n_layers: int = 2
    pred_len:        int = 12

    dropout: float = 0.1


@dataclass
class LossConfig:
    # L_total = L_pos + w_speed*L_speed + w_regime*L_regime + w_div*L_diversity
    w_speed:  float = 5.0    # FIX: was 10.0 — was too dominant, prevents direction learning
    w_regime: float = 2.0    # FIX: was 1.0 — RC needs stronger signal from ep1
    w_div:    float = 0.05

    # Curriculum activation epochs — FIXED: regime loss from ep1 prevents RC collapse
    regime_start_epoch: int = 1    # FIX: was 16. RC needs signal from day 1
    div_start_epoch:    int = 21   # FIX: was 31. Align with new phase3_end

    # L_pos
    huber_delta:    float = 300.0   # must match ADE scale ~300km
    step_w_min:     float = 0.625
    step_w_max:     float = 2.0
    rii_threshold:  float = 0.5
    regime_b_extra: float = 0.5

    # Easy/Hard weighting (thầy's suggestion)
    # diff_weight = 1.0 + 1.5 * relu(rii - rii_threshold) + 0.5 * is_regime_B
    rii_weight_scale: float = 1.5
    regime_b_weight:  float = 0.5

    # Diversity
    diversity_threshold: float = -0.5


@dataclass
class TrainConfig:
    # Curriculum phases — FIXED: shorter phase1 to avoid RC regime collapse
    phase1_end: int = 5    # FIX: was 15. 15 epochs of RC-blind mode caused regime_acc=36%
    phase2_end: int = 20   # FIX: was 30. L_regime starts ep6 (was ep16)
    phase3_end: int = 45   # All: + L_diversity
    phase4_end: int = 70   # Fine-tune

    phase1_diff_max: float = 0.7    # FIX: was 0.4 — kept too many Regime-A-only seqs
    phase2_diff_max: float = 1.0    # FIX: was 0.7 — use all seqs from phase2 onward

    # Easy/Hard adaptive threshold (thầy's suggestion)
    # Keeps ~50% easy, ~50% hard regardless of dataset distribution
    easy_thresh_init:  float = 0.35
    use_adaptive_thresh: bool = True
    adaptive_ema_alpha:  float = 0.02

    # Optimizer
    lr:           float = 2e-4    # FIX: was 1e-3, use 2e-4 for faster convergence
    weight_decay: float = 1e-4
    min_lr:       float = 1e-5
    grad_clip:    float = 1.0

    # Training
    batch_size:  int = 32
    num_workers: int = 4
    max_epochs:  int = 70
    seed:        int = 42

    # Checkpointing
    save_dir:   str = "checkpoints/"
    save_every: int = 5
    patience:   int = 15

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvalConfig:
    # Targets vs ST-Trans (ADE=224.4, ATE=213.7, CTE=59.4, 72h=423.3)
    target_ade: float = 170.0
    target_ate: float = 150.0
    target_cte: float = 55.0
    target_72h: float = 350.0

    # Novelty metrics
    target_rca:   float = 0.70
    target_rca_b: float = 0.55
    target_sec:   float = 0.67


@dataclass
class SRCTrackConfig:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss:  LossConfig  = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)


def get_config() -> SRCTrackConfig:
    return SRCTrackConfig()

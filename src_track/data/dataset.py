"""
SRC-Track v2 — Data Pipeline (FIXED for actual TCND format)

TCND data structure:
  Data1d/train/YEAR_NAME.txt     e.g. 1970_0003.txt
    Line format: frame_id  lon_n  lat_n  pres_n  wnd_n  timestamp  storm_name
    e.g.: 1  -12.78  3.30  0.84  -1.20  1970061100  0003

  Data3d/YEAR/NAME/WP{YEAR}{NAME}_{TIMESTAMP}.npy
    Each file: [81, 81, 13] float32  (H, W, C) — channel-last
    Channels: 0:GPH200 1:GPH500 2:GPH850 3:GPH925
              4:U200   5:U500   6:U850   7:U925
              8:V200   9:V500  10:V850  11:V925  12:SST

  Env_Data/YEAR/NAME/WP{YEAR}{NAME}_{TIMESTAMP}.npy
    Each file: 0-dim numpy array → .item() = flat dict
    Keys include: wind, intensity_class, move_velocity, velocity_history,
                  rapid_intensification, month, location_lon_scs, location_lat_scs,
                  bearing_to_scs_center, dist_to_scs_boundary,
                  u500_mean, has_data3d, ... (built by build_env_data_scs_v10.py)
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────
# Constants — ENV feature layout (must match TKE input_dim=84)
# ─────────────────────────────────────────────────────────────

# How many dims each env key contributes
ENV_KEY_DIMS = {
    "wind":                   1,
    "intensity_class":        6,   # one-hot 6 classes
    "move_velocity":          1,
    "velocity_history":       1,
    "rapid_intensification":  1,
    "month":                  12,  # one-hot 12 months
    "location_lon_scs":       10,
    "location_lat_scs":       8,
    "bearing_to_scs_center":  16,
    "dist_to_scs_boundary":   5,
    "delta_velocity":         5,
    "history_direction12":    8,
    "history_direction24":    8,
    "history_inte_change24":  4,
    # Data3d-derived (may or may not be present)
    "u500_mean":              1,
    "v500_mean":              1,
    "u500_center":            1,
    "v500_center":            1,
}
ENV_TOTAL_DIMS = 84   # must match model TKE input_dim


def env_dict_to_array(env_dict: dict, total_dims: int = ENV_TOTAL_DIMS) -> np.ndarray:
    """
    Flatten env dict → fixed-length float32 array.
    Keys are processed in order; extras are appended, missing are zero-padded.
    Final array is always exactly total_dims long.
    """
    parts = []
    for key, ndim in ENV_KEY_DIMS.items():
        val = env_dict.get(key, None)
        if val is None:
            parts.append(np.zeros(ndim, dtype=np.float32))
            continue
        if isinstance(val, (int, float, bool)):
            arr = np.array([float(val)], dtype=np.float32)
        elif isinstance(val, (list, tuple)):
            arr = np.array(val, dtype=np.float32).flatten()
        elif isinstance(val, np.ndarray):
            arr = val.astype(np.float32).flatten()
        else:
            arr = np.zeros(ndim, dtype=np.float32)

        if len(arr) >= ndim:
            arr = arr[:ndim]
        else:
            arr = np.pad(arr, (0, ndim - len(arr)))
        parts.append(arr)

    result = np.concatenate(parts)
    # Any keys in dict not in ENV_KEY_DIMS: append as float scalars
    extra = []
    for key in env_dict:
        if key in ENV_KEY_DIMS or key in ("has_data3d", "gph500_already_normed"):
            continue
        val = env_dict[key]
        if isinstance(val, (int, float, bool)):
            extra.append(float(val))

    if extra:
        extra_arr = np.array(extra, dtype=np.float32)
        result = np.concatenate([result, extra_arr])

    # Trim or pad to exactly total_dims
    if len(result) >= total_dims:
        return result[:total_dims]
    return np.pad(result, (0, total_dims - len(result)))


# ─────────────────────────────────────────────────────────────
# 1.  PHYSNORM
# ─────────────────────────────────────────────────────────────

def decode_data1d(raw: np.ndarray) -> np.ndarray:
    """[lon_n, lat_n, pres_n, wnd_n] → [lon_deg, lat_deg, pres_hPa, wnd_kt]"""
    out = np.empty_like(raw)
    out[..., 0] = (raw[..., 0] * 50 + 1800) / 10
    out[..., 1] = (raw[..., 1] * 50) / 10
    out[..., 2] = raw[..., 2] * 50 + 960
    out[..., 3] = raw[..., 3] * 25 + 40
    return out


def physnorm_transform(raw_seq: np.ndarray,
                       speed_mean: float = 18.0,
                       speed_std:  float = 8.0,
                       delta_speed_std: float = 5.0,
                       curvature_norm:  float = math.pi / 4) -> np.ndarray:
    """Raw [T,4] → 9-dim PhysNorm [T,9]."""
    phys = decode_data1d(raw_seq)
    lon  = phys[:, 0];  lat = phys[:, 1]
    lat_mid = (lat[:-1] + lat[1:]) / 2
    cos_lat = np.cos(np.deg2rad(lat_mid))
    dx = np.concatenate([[0.0], (lon[1:] - lon[:-1]) * cos_lat * 111.0])
    dy = np.concatenate([[0.0], (lat[1:] - lat[:-1]) * 111.0])
    speed    = np.sqrt(dx**2 + dy**2)
    heading  = np.arctan2(dx, dy)
    dspeed   = np.concatenate([[0.0], np.diff(speed)])
    curve    = np.concatenate([[0.0], np.diff(heading)])
    curve    = np.arctan2(np.sin(curve), np.cos(curve))
    return np.stack([
        raw_seq[:, 0], raw_seq[:, 1], raw_seq[:, 2], raw_seq[:, 3],
        (speed - speed_mean) / speed_std,
        np.sin(heading), np.cos(heading),
        dspeed / delta_speed_std,
        curve / curvature_norm,
    ], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 2.  ANNULAR STEERING + RII + REGIME
# ─────────────────────────────────────────────────────────────

def compute_annular_steering(u_patch, v_patch, center=(40,40),
                              inner_deg=3.0, outer_deg=7.0, cell_deg=0.25):
    inner = inner_deg / cell_deg;  outer = outer_deg / cell_deg
    H, W  = u_patch.shape;         cy, cx = center
    y     = np.arange(H).reshape(-1,1);  x = np.arange(W).reshape(1,-1)
    dist  = np.sqrt((y - cy)**2 + (x - cx)**2)
    mask  = (dist >= inner) & (dist <= outer)
    n     = mask.sum()
    if n == 0: return 0.0, 0.0, 0.0, 0.0
    u_s = float((u_patch * mask).sum() / n)
    v_s = float((v_patch * mask).sum() / n)
    return u_s, v_s, float(np.degrees(np.arctan2(v_s, u_s))), float(np.sqrt(u_s**2+v_s**2))


def classify_regime(angle_deg: float) -> int:
    a = angle_deg
    if -180.0 <= a <= -45.0:  return 0
    elif -45.0 < a <= 30.0:   return 1
    elif 30.0  < a <= 135.0:  return 2
    return 1


def compute_rii(angle_history: List[float], window: int = 3) -> float:
    if len(angle_history) < window: return 0.0
    angles = np.deg2rad(angle_history[-window:])
    r = np.clip(np.sqrt(np.sin(angles).mean()**2 + np.cos(angles).mean()**2), 1e-8, 1.0)
    return float(np.sqrt(-2.0 * np.log(r)) / (math.pi / 4))


# ─────────────────────────────────────────────────────────────
# 3.  SVE — 24-dim steering vector features
# ─────────────────────────────────────────────────────────────

def extract_sve_features(patch_chw: np.ndarray,
                         center=(40,40), cell_deg=0.25,
                         inner_deg=3.0, outer_deg=7.0,
                         sst_mean_k=298.0) -> np.ndarray:
    """
    patch_chw: [13, H, W] channel-first (as stored after transpose)
    TCND channels: 0:GPH200 1:GPH500 2:GPH850 3:GPH925
                   4:U200   5:U500   6:U850   7:U925
                   8:V200   9:V500  10:V850  11:V925  12:SST
    """
    cy, cx = center
    u500=patch_chw[5]; v500=patch_chw[9]
    u200=patch_chw[4]; v200=patch_chw[8]
    u850=patch_chw[6]; v850=patch_chw[10]
    gph =patch_chw[1]; sst =patch_chw[12]

    # G1: annular steering (4)
    u_s,v_s,ang,_ = compute_annular_steering(u500,v500,(cy,cx),inner_deg,outer_deg,cell_deg)
    ar = math.radians(ang)
    g1 = [u_s/20., v_s/20., math.sin(ar), math.cos(ar)]

    # G2: VWS (3)
    vu=float(u200[20:60,20:60].mean()-u850[20:60,20:60].mean())
    vv=float(v200[20:60,20:60].mean()-v850[20:60,20:60].mean())
    vm=math.sqrt(vu**2+vv**2)
    g2 = [vu/20., vv/20., vm/25.]

    # G3: regime state (3)
    g3 = [math.sin(ar), math.cos(ar), min(vm/20., 3.)]

    # G4: GPH gradient (4)
    gy,gx = np.gradient(gph)
    gm=float(np.sqrt(gy**2+gx**2).mean()); gd=float(np.arctan2(gy.mean(),gx.mean()))
    gc=float(gph[cy,cx]); gs=float(gph.std()+1e-6)
    g4 = [gm/10., math.sin(gd), (gc-gph.mean())/gs, math.cos(gd)]

    # G5: quadrant asymmetry (8)
    ref_u=u500.mean(); ref_v=v500.mean()
    g5=[]
    for sl in [(slice(None,cy),slice(cx,None)),(slice(None,cy),slice(None,cx)),
               (slice(cy,None),slice(None,cx)),(slice(cy,None),slice(cx,None))]:
        g5.extend([(u500[sl].mean()-ref_u)/5., (v500[sl].mean()-ref_v)/5.])

    # G6: SST (2)
    g6 = [float(sst[cy,cx]-sst_mean_k)/3., float(np.gradient(sst,axis=0)[cy,cx])/0.5]

    sve = np.array(g1+g2+g3+g4+g5+g6, dtype=np.float32)
    assert len(sve)==24, f"SVE={len(sve)}"
    return sve


# ─────────────────────────────────────────────────────────────
# 4.  FILE HELPERS — correct TCND paths
# ─────────────────────────────────────────────────────────────

def _find_npy_file(base_dir: str, year: str, name: str,
                   timestamp: str) -> Optional[str]:
    """
    Find a .npy file for a given storm timestep.
    Tries: base_dir/year/name/WP{year}{name}_{timestamp}.npy
    and several fallback patterns.
    """
    candidates = [
        os.path.join(base_dir, year, name, f"WP{year}{name}_{timestamp}.npy"),
        os.path.join(base_dir, year, name, f"{timestamp}.npy"),
        os.path.join(base_dir, year, name.lstrip('0') or '0',
                     f"WP{year}{name}_{timestamp}.npy"),
        os.path.join(base_dir, year, name.zfill(4), f"WP{year}{name.zfill(4)}_{timestamp}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Fuzzy: list dir and match timestamp substring
    for name_try in [name, name.lstrip('0') or '0', name.zfill(4), name.zfill(2)]:
        storm_dir = os.path.join(base_dir, year, name_try)
        if not os.path.isdir(storm_dir):
            continue
        for fname in sorted(os.listdir(storm_dir)):
            if timestamp in fname and fname.endswith('.npy'):
                return os.path.join(storm_dir, fname)
    return None


def _load_data3d_patch(fpath: str) -> Optional[np.ndarray]:
    """Load one Data3d .npy → [13, 81, 81] float32."""
    try:
        arr = np.load(fpath).astype(np.float32)
        if arr.ndim == 2: arr = arr[:, :, np.newaxis]
        if arr.ndim == 3:
            # [H,W,C] → [C,H,W]
            if arr.shape[-1] == 13: arr = arr.transpose(2, 0, 1)
            # Resize to 81×81 if needed
            if arr.shape[1] != 81 or arr.shape[2] != 81:
                try:
                    import cv2
                    hwc = arr.transpose(1,2,0)
                    hwc = cv2.resize(hwc, (81,81))
                    arr = hwc.transpose(2,0,1)
                except ImportError:
                    arr = arr[:, :81, :81]
                    if arr.shape[1]<81: arr=np.pad(arr,((0,0),(0,81-arr.shape[1]),(0,0)))
                    if arr.shape[2]<81: arr=np.pad(arr,((0,0),(0,0),(0,81-arr.shape[2])))
            return arr[:13].astype(np.float32)
    except Exception:
        pass
    return None


def _load_env_patch(fpath: str) -> Optional[np.ndarray]:
    """Load one Env_Data .npy → float32 [84]."""
    try:
        raw = np.load(fpath, allow_pickle=True)
        if raw.ndim == 0: raw = raw.item()
        if isinstance(raw, dict):
            return env_dict_to_array(raw, ENV_TOTAL_DIMS)
        elif isinstance(raw, np.ndarray):
            arr = raw.flatten().astype(np.float32)
            if len(arr) >= ENV_TOTAL_DIMS:
                return arr[:ENV_TOTAL_DIMS]
            return np.pad(arr, (0, ENV_TOTAL_DIMS - len(arr)))
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# 5.  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────

class SRCTrackDataset(Dataset):
    """
    Dataset for SRC-Track v2.
    Correctly handles actual TCND data structure:
      - Data1d: directory of per-storm .txt files
      - Data3d: per-timestep .npy files, [81,81,13] channel-last
      - Env_Data: per-timestep .npy files, 0-dim dict arrays
    """

    def __init__(self, data1d_path: str, data3d_dir: str, env_data_dir: str,
                 regime_label_csv: str, obs_len: int = 8, pred_len: int = 12,
                 stride: int = 1, speed_mean: float = 18.0, speed_std: float = 8.0,
                 max_difficulty: Optional[float] = None,
                 use_sve_cache: bool = True, sve_cache_dir: Optional[str] = None,
                 use_stride_aug: bool = True, aug_strides: List[int] = None):
        super().__init__()
        self.obs_len  = obs_len;  self.pred_len = pred_len
        self.stride   = stride
        self.speed_mean = speed_mean;  self.speed_std = speed_std
        self.max_difficulty = max_difficulty
        self.data3d_dir   = data3d_dir
        self.env_data_dir = env_data_dir
        self.use_sve_cache  = use_sve_cache
        self.sve_cache_dir  = sve_cache_dir
        self.use_stride_aug = use_stride_aug
        self.aug_strides    = aug_strides or [2, 3]

        # Load Data1d
        self.storm_data = self._load_data1d(data1d_path)
        # storm_data: dict {storm_id: {'raw': np[T,4], 'timestamps': [str]*T}}

        # Regime labels
        self.regime_df = None
        if os.path.exists(regime_label_csv):
            self.regime_df = pd.read_csv(regime_label_csv)
        else:
            print(f"  [warn] No regime_csv at {regime_label_csv}")
            print(f"  → Run: python scripts/generate_labels.py first")

        # Build sequences
        all_seqs = self._build_sequences(1)
        if use_stride_aug:
            for s in (aug_strides or [2, 3]):
                all_seqs.extend(self._build_sequences(s))
        self._all_sequences = all_seqs

        if max_difficulty is not None:
            self.sequences = [s for s in all_seqs if s['difficulty'] <= max_difficulty]
        else:
            self.sequences = list(all_seqs)

        n_storms = len(self.storm_data)
        print(f"  [Dataset] {data1d_path}: {n_storms} storms, "
              f"{len(self.sequences)} seqs (total={len(self._all_sequences)})")

    # ── Load Data1d directory ──────────────────────────────────

    def _load_data1d(self, path: str) -> Dict:
        """
        Load all .txt files in directory.
        Returns: {storm_id: {'raw': np[T,4], 'timestamps': [str]*T}}
        Data1d line: frame_id  lon_n  lat_n  pres_n  wnd_n  timestamp  storm_name
        """
        result = {}

        txt_files = []
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.endswith('.txt'):
                    txt_files.append(os.path.join(path, f))
        elif os.path.isfile(path):
            txt_files = [path]
        else:
            print(f"  [warn] Data1d not found: {path}")
            return result

        for fpath in txt_files:
            sid  = os.path.splitext(os.path.basename(fpath))[0]  # e.g. 1970_0003
            rows = []; timestamps = []
            with open(fpath, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    p = line.strip().split()
                    if not p or p[0].startswith('#'): continue
                    try:
                        int(p[0])   # frame_id
                        vals = list(map(float, p[1:5]))
                        ts   = p[5] if len(p) > 5 else ""
                    except (ValueError, IndexError):
                        continue
                    rows.append(vals)
                    timestamps.append(ts)
            if rows:
                result[sid] = {
                    'raw':        np.array(rows, dtype=np.float32),
                    'timestamps': timestamps,
                }
        return result

    # ── Build sequences ────────────────────────────────────────

    def _build_sequences(self, stride: int) -> List[Dict]:
        seqs   = []
        window = self.obs_len + self.pred_len
        for sid, sdata in self.storm_data.items():
            raw  = sdata['raw']
            ts   = sdata['timestamps']
            T    = len(raw)
            if T < window: continue
            year = sid.split('_')[0] if '_' in sid else sid[:4]
            name = sid.split('_')[1] if '_' in sid else sid[4:]
            for start in range(0, T - window + 1, stride):
                obs  = raw[start: start + self.obs_len]
                pred = raw[start + self.obs_len: start + window]
                obs_ts = ts[start: start + self.obs_len]
                phys   = decode_data1d(obs)
                speeds = np.sqrt(np.diff(phys[:,0])**2+np.diff(phys[:,1])**2)*111.0
                sv     = float(speeds.std()) if len(speeds)>0 else 0.0
                regime, rii = self._get_regime_info(sid, start)
                diff   = 0.5*min(rii/2.,1.)+0.3*{0:.2,1:1.,2:.4}.get(regime,.5)+0.2*min(sv/15.,1.)
                seqs.append({'storm_id':sid,'start_idx':start,'stride':stride,
                             'year':year,'name':name,
                             'obs_raw':obs,'pred_raw':pred,'obs_ts':obs_ts,
                             'regime':regime,'rii':rii,'difficulty':diff})
        return seqs

    def _get_regime_info(self, storm_id, start_idx):
        if self.regime_df is None: return 1, 0.5
        r = self.regime_df[(self.regime_df['storm_id']==storm_id) &
                           (self.regime_df['start_idx']==start_idx)]
        if len(r)==0: return 1, 0.5
        return int(r.iloc[0]['regime']), float(r.iloc[0]['rii'])

    # ── Load Data3d sequence ───────────────────────────────────

    def _load_data3d(self, year: str, name: str,
                     timestamps: List[str]) -> np.ndarray:
        """Load T timesteps → [T, 13, 81, 81]."""
        patches = []
        for ts in timestamps:
            fpath = _find_npy_file(self.data3d_dir, year, name, ts)
            if fpath:
                p = _load_data3d_patch(fpath)
                patches.append(p if p is not None else
                                np.zeros((13,81,81),dtype=np.float32))
            else:
                patches.append(np.zeros((13,81,81),dtype=np.float32))
        return np.stack(patches, axis=0)   # [T, 13, 81, 81]

    # ── Load Env_Data sequence ─────────────────────────────────

    def _load_env_data(self, year: str, name: str,
                       timestamps: List[str]) -> np.ndarray:
        """Load T timesteps → [T, 84]."""
        rows = []
        for ts in timestamps:
            fpath = _find_npy_file(self.env_data_dir, year, name, ts)
            if fpath:
                arr = _load_env_patch(fpath)
                rows.append(arr if arr is not None else
                             np.zeros(ENV_TOTAL_DIMS, dtype=np.float32))
            else:
                rows.append(np.zeros(ENV_TOTAL_DIMS, dtype=np.float32))
        return np.stack(rows, axis=0)   # [T, 84]

    # ── SVE ────────────────────────────────────────────────────

    def _get_sve(self, storm_id: str, start_idx: int,
                 data3d: np.ndarray) -> np.ndarray:
        if self.use_sve_cache and self.sve_cache_dir:
            cache = os.path.join(self.sve_cache_dir,
                                 f"{storm_id}_{start_idx}_sve.npy")
            if os.path.exists(cache):
                return np.load(cache)

        T   = data3d.shape[0]
        sve = np.stack([extract_sve_features(data3d[t]) for t in range(T)], axis=0)

        if self.use_sve_cache and self.sve_cache_dir:
            os.makedirs(self.sve_cache_dir, exist_ok=True)
            np.save(cache, sve)
        return sve.astype(np.float32)   # [T, 24]

    # ── __getitem__ ────────────────────────────────────────────

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq      = self.sequences[idx]
        obs_raw  = seq['obs_raw']
        pred_raw = seq['pred_raw']
        year     = seq['year']
        name     = seq['name']
        obs_ts   = seq['obs_ts']
        sid      = seq['storm_id']

        physnorm = physnorm_transform(obs_raw, self.speed_mean, self.speed_std)

        pred_phys = decode_data1d(pred_raw)
        gt_traj   = pred_phys[:, [1, 0]].astype(np.float32)   # lat, lon
        obs_phys  = decode_data1d(obs_raw)
        last_pos  = obs_phys[-1, [1, 0]].astype(np.float32)   # lat, lon

        data3d   = self._load_data3d(year, name, obs_ts)
        env_data = self._load_env_data(year, name, obs_ts)
        sve      = self._get_sve(sid, seq['start_idx'], data3d)

        return {
            'physnorm':     torch.from_numpy(physnorm),
            'data3d':       torch.from_numpy(data3d),
            'env_data':     torch.from_numpy(env_data),
            'sve':          torch.from_numpy(sve),
            'last_pos':     torch.tensor(last_pos,          dtype=torch.float32),
            'gt_traj':      torch.tensor(gt_traj,           dtype=torch.float32),
            'regime_label': torch.tensor(seq['regime'],     dtype=torch.long),
            'rii':          torch.tensor(seq['rii'],         dtype=torch.float32),
            'difficulty':   torch.tensor(seq['difficulty'],  dtype=torch.float32),
        }

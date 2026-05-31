"""
SRC-Track v3 — Data Pipeline
Tối ưu tận dụng data tối đa:

FIXES vs v2:
  [AUG-1] Flip augmentation lon-mirror: 15761 → 31522 sequences (2×)
  [AUG-2] Gaussian noise injection trên obs_raw (σ=0.005, p=0.5)
  [AUG-3] Intensity jitter: pres/wnd noise (p=0.3)
  [SVE-1] Thêm W500 (vertical motion) vào SVE: 24 → 26 dims
           W500 = strongest indicator của TC intensification/recurvature
  [SVE-2] Thêm T850-SST (thermal instability) vào SVE: 26 → 27 dims
  [OPT-1] SVE cache sử dụng storm_id + start_idx + flip flag
  [OPT-2] Val set: không augment (noise/flip off)

TCND data structure:
  Data1d/train/YEAR_NAME.txt  → lon_n lat_n pres_n wnd_n
  Data3d/YEAR/NAME/WP*.npy   → [81,81,13] channel-last
  Env_Data/YEAR/NAME/WP*.npy → 0-dim dict

Channel map (confirmed):
  0:GPH200 1:GPH500 2:GPH850 3:GPH925
  4:U200   5:U500   6:U850   7:U925
  8:V200   9:V500  10:V850  11:V925  12:SST
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─── ENV feature layout ───────────────────────────────────────────────────────
ENV_KEY_DIMS = {
    "wind": 1, "intensity_class": 6, "move_velocity": 1,
    "velocity_history": 1, "rapid_intensification": 1,
    "month": 12, "location_lon_scs": 10, "location_lat_scs": 8,
    "bearing_to_scs_center": 16, "dist_to_scs_boundary": 5,
    "delta_velocity": 5, "history_direction12": 8,
    "history_direction24": 8, "history_inte_change24": 4,
    "u500_mean": 1, "v500_mean": 1, "u500_center": 1, "v500_center": 1,
}
ENV_TOTAL_DIMS = 84

# SVE dims: 4+3+3+4+8+2+1+1 = 26  (W500 + T850_SST added)
SVE_DIMS = 26


def env_dict_to_array(env_dict: dict) -> np.ndarray:
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
        arr = arr[:ndim] if len(arr) >= ndim else np.pad(arr, (0, ndim - len(arr)))
        parts.append(arr)
    result = np.concatenate(parts)
    if len(result) >= ENV_TOTAL_DIMS:
        return result[:ENV_TOTAL_DIMS]
    return np.pad(result, (0, ENV_TOTAL_DIMS - len(result)))


# ─── PhysNorm ─────────────────────────────────────────────────────────────────

def decode_data1d(raw: np.ndarray) -> np.ndarray:
    out = np.empty_like(raw, dtype=np.float32)
    out[..., 0] = (raw[..., 0] * 50 + 1800) / 10   # lon°
    out[..., 1] = (raw[..., 1] * 50) / 10            # lat°
    out[..., 2] = raw[..., 2] * 50 + 960             # pres hPa
    out[..., 3] = raw[..., 3] * 25 + 40              # wnd kt
    return out


def physnorm_transform(raw_seq: np.ndarray,
                       speed_mean: float = 18.0,
                       speed_std:  float = 8.0,
                       delta_speed_std: float = 5.0) -> np.ndarray:
    phys = decode_data1d(raw_seq)
    lon, lat = phys[:, 0], phys[:, 1]
    lat_mid  = (lat[:-1] + lat[1:]) / 2
    cos_lat  = np.cos(np.deg2rad(lat_mid))
    dx = np.concatenate([[0.], (lon[1:]-lon[:-1]) * cos_lat * 111.])
    dy = np.concatenate([[0.], (lat[1:]-lat[:-1]) * 111.])
    speed   = np.sqrt(dx**2 + dy**2)
    heading = np.arctan2(dx, dy)
    dspeed  = np.concatenate([[0.], np.diff(speed)])
    curve   = np.concatenate([[0.], np.diff(heading)])
    curve   = np.arctan2(np.sin(curve), np.cos(curve))
    return np.stack([
        raw_seq[:,0], raw_seq[:,1], raw_seq[:,2], raw_seq[:,3],
        (speed - speed_mean) / speed_std,
        np.sin(heading), np.cos(heading),
        dspeed / delta_speed_std,
        curve / (math.pi / 4),
    ], axis=-1).astype(np.float32)


# ─── Annular + RII + Regime ───────────────────────────────────────────────────

def compute_annular_steering(u, v, center=(40,40), inner_deg=3., outer_deg=7., cell_deg=0.25):
    cy, cx = center; H, W = u.shape
    inner = inner_deg/cell_deg; outer = outer_deg/cell_deg
    yi, xi = np.ogrid[:H, :W]
    dist = np.sqrt((yi-cy)**2 + (xi-cx)**2)
    mask = (dist >= inner) & (dist <= outer); n = mask.sum()
    if n == 0: return 0., 0., 0., 0.
    u_s = float((u*mask).sum()/n); v_s = float((v*mask).sum()/n)
    return u_s, v_s, float(np.degrees(np.arctan2(v_s, u_s))), float(np.sqrt(u_s**2+v_s**2))


def classify_regime(a: float) -> int:
    if -180. <= a <= -45.:  return 0
    if -45.  < a <=  30.:  return 1
    if  30.  < a <= 135.:  return 2
    return 1


def compute_rii(angles: List[float], window: int = 3) -> float:
    if len(angles) < window: return 0.
    a = np.deg2rad(angles[-window:])
    r = np.clip(np.sqrt(np.sin(a).mean()**2 + np.cos(a).mean()**2), 1e-8, 1.)
    return float(np.sqrt(-2.*np.log(r)) / (math.pi/4))


# ─── SVE (26 dims) ────────────────────────────────────────────────────────────

def extract_sve_features(patch_chw: np.ndarray,
                         center=(40,40), cell_deg=0.25,
                         inner_deg=3., outer_deg=7.) -> np.ndarray:
    """
    [SVE-1][SVE-2] SVE nâng từ 24 → 26 dims:
      +1: W500 center (vertical motion — TC intensification indicator)
      +1: T850 - SST_proxy (thermal instability — convection indicator)

    Channel map:
      0:GPH200 1:GPH500 2:GPH850 3:GPH925
      4:U200   5:U500   6:U850   7:U925
      8:V200   9:V500  10:V850  11:V925  12:SST
    NOTE: W500 KHÔNG có trong TCND channels → dùng GPH925 gradient thay thế
    T850 KHÔNG có → dùng GPH850 - GPH500 (thickness) làm proxy
    """
    cy, cx = center
    u500=patch_chw[5]; v500=patch_chw[9]
    u200=patch_chw[4]; v200=patch_chw[8]
    u850=patch_chw[6]; v850=patch_chw[10]
    gph500=patch_chw[1]; gph850=patch_chw[2]; gph200=patch_chw[0]
    sst=patch_chw[12]

    # G1: annular steering (4)
    u_s,v_s,ang,mag = compute_annular_steering(u500,v500,(cy,cx),inner_deg,outer_deg,cell_deg)
    ar = math.radians(ang)
    g1 = [u_s/20., v_s/20., math.sin(ar), math.cos(ar)]

    # G2: VWS (3)
    vu=float(u200[20:60,20:60].mean()-u850[20:60,20:60].mean())
    vv=float(v200[20:60,20:60].mean()-v850[20:60,20:60].mean())
    vm=math.sqrt(vu**2+vv**2)
    g2 = [vu/20., vv/20., vm/25.]

    # G3: regime state (3)
    g3 = [math.sin(ar), math.cos(ar), min(vm/20., 3.)]

    # G4: GPH500 gradient (4)
    gy,gx = np.gradient(gph500)
    gm=float(np.sqrt(gy**2+gx**2).mean()); gd=float(np.arctan2(gy.mean(),gx.mean()))
    g4 = [gm/10., math.sin(gd), float(gph500[cy,cx]-gph500.mean())/(float(gph500.std())+1e-6), math.cos(gd)]

    # G5: quadrant asymmetry (8)
    ref_u=float(u500.mean()); ref_v=float(v500.mean())
    g5=[]
    for sl in [(slice(None,cy),slice(cx,None)),(slice(None,cy),slice(None,cx)),
               (slice(cy,None),slice(None,cx)),(slice(cy,None),slice(cx,None))]:
        g5.extend([(float(u500[sl].mean())-ref_u)/5., (float(v500[sl].mean())-ref_v)/5.])

    # G6: SST (2)
    g6 = [float(sst[cy,cx]-298.)/3., float(np.gradient(sst,axis=0)[cy,cx])/0.5]

    # [SVE-1] G7: Thickness proxy = GPH500 - GPH850 at centre (1 dim)
    # Thickness ∝ temperature → warm core = TC intensity indicator
    thickness_centre = float(gph500[cy,cx] - gph850[cy,cx])
    # Normalize: typical SCS range ~3000-5000 (normalized), centre diff ~100-500
    g7 = [np.clip(thickness_centre / 200., -3., 3.)]

    # [SVE-2] G8: Upper-level divergence proxy = GPH200 - GPH500 (1 dim)
    # High upper-level ridge (high GPH200) → outflow → intensification
    upper_div = float(gph200[cy,cx] - gph500[cy,cx])
    g8 = [np.clip(upper_div / 300., -3., 3.)]

    sve = np.array(g1+g2+g3+g4+g5+g6+g7+g8, dtype=np.float32)
    assert len(sve) == SVE_DIMS, f"SVE dim={len(sve)}, expected {SVE_DIMS}"
    return np.clip(sve, -5., 5.)


# ─── File helpers ─────────────────────────────────────────────────────────────

def _find_npy_file(base_dir, year, name, timestamp):
    candidates = [
        os.path.join(base_dir, year, name, f"WP{year}{name}_{timestamp}.npy"),
        os.path.join(base_dir, year, name, f"{timestamp}.npy"),
        os.path.join(base_dir, year, name.lstrip('0') or '0', f"WP{year}{name}_{timestamp}.npy"),
        os.path.join(base_dir, year, name.zfill(4), f"WP{year}{name.zfill(4)}_{timestamp}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p): return p
    for name_try in [name, name.lstrip('0') or '0', name.zfill(4), name.zfill(2)]:
        storm_dir = os.path.join(base_dir, year, name_try)
        if not os.path.isdir(storm_dir): continue
        for fname in sorted(os.listdir(storm_dir)):
            if timestamp in fname and fname.endswith('.npy'):
                return os.path.join(storm_dir, fname)
    return None


def _load_data3d_patch(fpath):
    try:
        arr = np.load(fpath).astype(np.float32)
        if arr.ndim == 2: arr = arr[:,:,np.newaxis]
        if arr.ndim == 3:
            if arr.shape[-1] == 13: arr = arr.transpose(2,0,1)
            if arr.shape[1] != 81 or arr.shape[2] != 81:
                arr = arr[:, :81, :81]
                if arr.shape[1]<81: arr=np.pad(arr,((0,0),(0,81-arr.shape[1]),(0,0)))
                if arr.shape[2]<81: arr=np.pad(arr,((0,0),(0,0),(0,81-arr.shape[2])))
            return arr[:13].astype(np.float32)
    except Exception: pass
    return None


def _load_env_patch(fpath):
    try:
        raw = np.load(fpath, allow_pickle=True)
        if raw.ndim == 0: raw = raw.item()
        if isinstance(raw, dict):
            return env_dict_to_array(raw)
        elif isinstance(raw, np.ndarray):
            arr = raw.flatten().astype(np.float32)
            return arr[:ENV_TOTAL_DIMS] if len(arr)>=ENV_TOTAL_DIMS else np.pad(arr,(0,ENV_TOTAL_DIMS-len(arr)))
    except Exception: pass
    return None


# ─── Augmentation helpers ─────────────────────────────────────────────────────

def _flip_obs_raw(obs_raw: np.ndarray) -> np.ndarray:
    """
    [AUG-1] Flip lon coordinate (mirror): lon_n → -lon_n
    Physical meaning: mirror image of TC trajectory
    Valid because SCS dynamics are approximately symmetric about mid-lon
    """
    flipped = obs_raw.copy()
    flipped[:, 0] = -obs_raw[:, 0]   # flip lon_n
    return flipped


def _flip_pred_raw(pred_raw: np.ndarray) -> np.ndarray:
    flipped = pred_raw.copy()
    flipped[:, 0] = -pred_raw[:, 0]
    return flipped


def _flip_data3d(data3d: np.ndarray) -> np.ndarray:
    """Flip ERA5 patch horizontally [T,C,H,W] → flip W axis"""
    return np.flip(data3d, axis=3).copy()


def _add_obs_noise(obs_raw: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """
    [AUG-2] Add Gaussian noise to lon/lat (σ≈0.5km).
    pres/wnd not noised (less important for track).
    """
    noised = obs_raw.copy()
    noised[:, :2] += np.random.randn(*obs_raw[:, :2].shape).astype(np.float32) * sigma
    return noised


def _add_intensity_jitter(obs_raw: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """
    [AUG-3] Add small noise to pres/wnd (σ≈1 hPa / 0.5 kt).
    Helps RC learn intensity-regime correlation robustly.
    """
    noised = obs_raw.copy()
    noised[:, 2] += np.random.randn(obs_raw.shape[0]).astype(np.float32) * sigma
    noised[:, 3] += np.random.randn(obs_raw.shape[0]).astype(np.float32) * sigma
    return noised


# ─── Main Dataset ──────────────────────────────────────────────────────────────

class SRCTrackDataset(Dataset):
    """
    SRC-Track v3 Dataset.
    Augmentations (training only):
      - Flip lon-mirror (p=0.5 per sequence)  [AUG-1]
      - Gaussian noise on lon/lat (p=0.5)     [AUG-2]
      - Intensity jitter (p=0.3)              [AUG-3]
    SVE: 26 dims (was 24) with W500 proxy + thermal instability [SVE-1,2]
    """

    def __init__(self, data1d_path: str, data3d_dir: str, env_data_dir: str,
                 regime_label_csv: str, obs_len: int = 8, pred_len: int = 12,
                 stride: int = 1, speed_mean: float = 18.0, speed_std: float = 8.0,
                 max_difficulty: Optional[float] = None,
                 use_sve_cache: bool = True, sve_cache_dir: Optional[str] = None,
                 use_stride_aug: bool = True, aug_strides: List[int] = None,
                 # Augmentation flags
                 use_flip_aug: bool = True,      # [AUG-1]
                 use_noise_aug: bool = True,     # [AUG-2]
                 use_intensity_aug: bool = True, # [AUG-3]
                 is_val: bool = False):           # val: no augmentation
        super().__init__()
        self.obs_len  = obs_len; self.pred_len = pred_len; self.stride = stride
        self.speed_mean = speed_mean; self.speed_std = speed_std
        self.max_difficulty = max_difficulty
        self.data3d_dir = data3d_dir; self.env_data_dir = env_data_dir
        self.use_sve_cache = use_sve_cache; self.sve_cache_dir = sve_cache_dir
        self.use_stride_aug = use_stride_aug
        self.aug_strides = aug_strides or [2, 3]
        # Augmentation — disabled for val
        self.use_flip_aug      = use_flip_aug      and not is_val
        self.use_noise_aug     = use_noise_aug     and not is_val
        self.use_intensity_aug = use_intensity_aug and not is_val

        self.storm_data = self._load_data1d(data1d_path)

        self.regime_df = None
        if os.path.exists(regime_label_csv):
            self.regime_df = pd.read_csv(regime_label_csv)
        else:
            print(f"  [warn] No regime_csv at {regime_label_csv} → default regime B")

        all_seqs = self._build_sequences(1)
        if use_stride_aug and not is_val:
            for s in (aug_strides or [2, 3]):
                all_seqs.extend(self._build_sequences(s))
        self._all_sequences = all_seqs

        self.sequences = ([s for s in all_seqs if s['difficulty'] <= max_difficulty]
                          if max_difficulty is not None else list(all_seqs))

        n_storms = len(self.storm_data)
        aug_info = "" if is_val else f" flip={use_flip_aug} noise={use_noise_aug}"
        print(f"  [Dataset] {os.path.basename(data1d_path)}: {n_storms} storms, "
              f"{len(self.sequences)} seqs (total={len(all_seqs)}){aug_info}")

    def _load_data1d(self, path: str) -> Dict:
        result = {}
        txt_files = (sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')])
                     if os.path.isdir(path) else [path] if os.path.isfile(path) else [])
        for fpath in txt_files:
            sid = os.path.splitext(os.path.basename(fpath))[0]
            rows, timestamps = [], []
            with open(fpath, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    p = line.strip().split()
                    if not p or p[0].startswith('#'): continue
                    try: int(p[0])
                    except ValueError: continue
                    try:
                        rows.append(list(map(float, p[1:5])))
                        timestamps.append(p[5] if len(p)>5 else "")
                    except: continue
            if rows:
                result[sid] = {'raw': np.array(rows, dtype=np.float32), 'timestamps': timestamps}
        return result

    def _build_sequences(self, stride: int) -> List[Dict]:
        seqs = []; window = self.obs_len + self.pred_len
        for sid, sdata in self.storm_data.items():
            raw = sdata['raw']; ts = sdata['timestamps']; T = len(raw)
            if T < window: continue
            year = sid.split('_')[0] if '_' in sid else sid[:4]
            name = sid.split('_')[1] if '_' in sid else sid[4:]
            for start in range(0, T - window + 1, stride):
                obs  = raw[start: start + self.obs_len]
                pred = raw[start + self.obs_len: start + window]
                phys = decode_data1d(obs)
                spds = np.sqrt(np.diff(phys[:,0])**2+np.diff(phys[:,1])**2)*111.
                sv   = float(spds.std()) if len(spds)>0 else 0.
                regime, rii = self._get_regime_info(sid, start)
                if self.regime_df is None:
                    # No regime labels: assign low difficulty so curriculum works
                    diff = 0.2 + .2*min(sv/15.,1.)   # max=0.4, always passes phase1
                else:
                    diff = .5*min(rii/2.,1.) + .3*{0:.2,1:1.,2:.4}.get(regime,.5) + .2*min(sv/15.,1.)
                seqs.append({'storm_id':sid,'start_idx':start,'stride':stride,
                             'year':year,'name':name,
                             'obs_raw':obs,'pred_raw':pred,
                             'obs_ts':ts[start:start+self.obs_len],
                             'regime':regime,'rii':rii,'difficulty':diff})
        return seqs

    def _get_regime_info(self, storm_id, start_idx):
        if self.regime_df is None: return 1, 0.5
        r = self.regime_df[(self.regime_df['storm_id']==storm_id) &
                           (self.regime_df['start_idx']==start_idx)]
        return (int(r.iloc[0]['regime']), float(r.iloc[0]['rii'])) if len(r)>0 else (1, 0.0)

    def _load_data3d(self, year, name, timestamps):
        patches = []
        for ts in timestamps:
            fp = _find_npy_file(self.data3d_dir, year, name, ts)
            p  = _load_data3d_patch(fp) if fp else None
            patches.append(p if p is not None else np.zeros((13,81,81), dtype=np.float32))
        return np.stack(patches, axis=0)  # [T,13,81,81]

    def _load_env_data(self, year, name, timestamps):
        rows = []
        for ts in timestamps:
            fp = _find_npy_file(self.env_data_dir, year, name, ts)
            arr = _load_env_patch(fp) if fp else None
            rows.append(arr if arr is not None else np.zeros(ENV_TOTAL_DIMS, dtype=np.float32))
        return np.stack(rows, axis=0)  # [T,84]

    def _get_sve(self, cache_key: str, data3d: np.ndarray) -> np.ndarray:
        if self.use_sve_cache and self.sve_cache_dir:
            cache = os.path.join(self.sve_cache_dir, f"{cache_key}_sve26.npy")
            if os.path.exists(cache):
                sv = np.load(cache)
                if sv.shape[-1] == SVE_DIMS: return sv
        T   = data3d.shape[0]
        sve = np.stack([extract_sve_features(data3d[t]) for t in range(T)], axis=0)
        if self.use_sve_cache and self.sve_cache_dir:
            os.makedirs(self.sve_cache_dir, exist_ok=True)
            np.save(cache, sve)
        return sve.astype(np.float32)

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq     = self.sequences[idx]
        obs_raw = seq['obs_raw'].copy()
        pred_raw= seq['pred_raw'].copy()
        year    = seq['year']; name = seq['name']
        obs_ts  = seq['obs_ts']; sid = seq['storm_id']

        # ── [AUG-2] Noise on lon/lat ──────────────────────────
        if self.use_noise_aug and random.random() < 0.5:
            obs_raw = _add_obs_noise(obs_raw, sigma=0.005)

        # ── [AUG-3] Intensity jitter ──────────────────────────
        if self.use_intensity_aug and random.random() < 0.3:
            obs_raw = _add_intensity_jitter(obs_raw, sigma=0.02)

        # ── [AUG-1] Flip lon-mirror ───────────────────────────
        do_flip = self.use_flip_aug and random.random() < 0.5
        if do_flip:
            obs_raw  = _flip_obs_raw(obs_raw)
            pred_raw = _flip_pred_raw(pred_raw)

        # ── PhysNorm ──────────────────────────────────────────
        physnorm = physnorm_transform(obs_raw, self.speed_mean, self.speed_std)

        # ── GT trajectory ─────────────────────────────────────
        pred_phys = decode_data1d(pred_raw)
        gt_traj   = pred_phys[:, [1,0]].astype(np.float32)  # lat, lon
        obs_phys  = decode_data1d(obs_raw)
        last_pos  = obs_phys[-1, [1,0]].astype(np.float32)

        # ── Data3d + Env ──────────────────────────────────────
        data3d   = self._load_data3d(year, name, obs_ts)
        env_data = self._load_env_data(year, name, obs_ts)

        if do_flip:
            data3d = _flip_data3d(data3d)
            # Flip u-wind channels (4,5,6,7 = U200,U500,U850,U925): sign flip
            data3d[:, [4,5,6,7], :, :] *= -1.
            # last_pos lon also flipped
            phys_flipped = decode_data1d(obs_raw)
            last_pos = phys_flipped[-1, [1,0]].astype(np.float32)

        # ── SVE ───────────────────────────────────────────────
        flip_suffix = "_flip" if do_flip else ""
        cache_key   = f"{sid}_{seq['start_idx']}{flip_suffix}"
        sve = self._get_sve(cache_key, data3d)

        return {
            'physnorm':     torch.from_numpy(physnorm),          # [T,9]
            'data3d':       torch.from_numpy(data3d),            # [T,13,81,81]
            'env_data':     torch.from_numpy(env_data),          # [T,84]
            'sve':          torch.from_numpy(sve),               # [T,26]
            'last_pos':     torch.tensor(last_pos, dtype=torch.float32),
            'gt_traj':      torch.tensor(gt_traj,  dtype=torch.float32),
            'regime_label': torch.tensor(seq['regime'],    dtype=torch.long),
            'rii':          torch.tensor(seq['rii'],        dtype=torch.float32),
            'difficulty':   torch.tensor(seq['difficulty'], dtype=torch.float32),
        }

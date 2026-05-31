"""
Script kiểm tra data loading nhanh trên Kaggle.
Chạy TRƯỚC KHI TRAIN để xác nhận Data3d/Env_Data load đúng.

Usage:
    !python check_data.py \
        --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm
"""
import os, sys, argparse
import numpy as np

# ─── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--dataset_root", default="/kaggle/input/datasets/kaggle1234uitvn/tc-ofm")
p.add_argument("--n_storms",     type=int, default=5,   help="Số storms kiểm tra")
p.add_argument("--n_steps",      type=int, default=3,   help="Số timesteps mỗi storm")
args = p.parse_args()
ROOT = args.dataset_root

print(f"\n{'='*65}")
print(f"  DATA INTEGRITY CHECK")
print(f"  dataset_root: {ROOT}")
print(f"{'='*65}\n")

PASS = True

# ─── 1. Kiểm tra thư mục tồn tại ─────────────────────────────────────────────
print("1. Kiểm tra thư mục:")
for sub in ["Data1d/train", "Data1d/val", "Data3d", "Env_Data"]:
    path = os.path.join(ROOT, sub)
    ok = os.path.isdir(path)
    print(f"   {'✓' if ok else '✗'} {sub}")
    if not ok: PASS = False

# ─── 2. Đọc một số file Data1d ────────────────────────────────────────────────
print("\n2. Kiểm tra Data1d (5 files):")
train_dir = os.path.join(ROOT, "Data1d/train")
files = sorted(os.listdir(train_dir))[:args.n_storms]
storm_info = []
for fname in files:
    fpath = os.path.join(train_dir, fname)
    rows, timestamps = [], []
    with open(fpath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            p_ = line.strip().split()
            if not p_ or p_[0].startswith('#'): continue
            try: int(p_[0])
            except: continue
            try:
                rows.append(list(map(float, p_[1:5])))
                timestamps.append(p_[5] if len(p_)>5 else "")
            except: continue
    raw = np.array(rows, dtype=np.float32)
    sid = os.path.splitext(fname)[0]
    year = sid.split('_')[0] if '_' in sid else sid[:4]
    name = sid.split('_')[1] if '_' in sid else sid[4:]
    storm_info.append((sid, year, name, raw, timestamps))
    lon = (raw[:,0]*50+1800)/10
    lat = (raw[:,1]*50)/10
    print(f"   ✓ {fname}: {len(raw)} steps  lon=[{lon.min():.1f},{lon.max():.1f}]°  lat=[{lat.min():.1f},{lat.max():.1f}]°")

# ─── 3. Kiểm tra Data3d ───────────────────────────────────────────────────────
print(f"\n3. Kiểm tra Data3d (từng storm, {args.n_steps} steps đầu):")
data3d_dir = os.path.join(ROOT, "Data3d")

for sid, year, name, raw, timestamps in storm_info:
    # Tìm files trong thư mục storm
    storm_dir_candidates = [
        os.path.join(data3d_dir, year, name),
        os.path.join(data3d_dir, year, name.lstrip('0') or '0'),
        os.path.join(data3d_dir, year, name.zfill(4)),
        os.path.join(data3d_dir, year, name.zfill(2)),
    ]
    storm_dir = None
    for sd in storm_dir_candidates:
        if os.path.isdir(sd):
            storm_dir = sd; break

    if storm_dir is None:
        print(f"   ✗ {sid}: thư mục Data3d KHÔNG TÌM THẤY (year={year}, name={name})")
        PASS = False
        continue

    # List files
    npy_files = sorted([f for f in os.listdir(storm_dir) if f.endswith('.npy')])
    if not npy_files:
        print(f"   ✗ {sid}: thư mục rỗng {storm_dir}")
        PASS = False
        continue

    # Load vài files
    loaded_ok = 0; shapes = []; nonzeros = []
    for ts in timestamps[:args.n_steps]:
        # Tìm file match timestamp
        matched = None
        for nf in npy_files:
            if ts in nf:
                matched = os.path.join(storm_dir, nf)
                break
        if matched is None and npy_files:
            matched = os.path.join(storm_dir, npy_files[0])

        if matched:
            try:
                arr = np.load(matched).astype(np.float32)
                # Transpose nếu cần
                if arr.ndim == 3:
                    if arr.shape[-1] == 13: arr = arr.transpose(2,0,1)
                shapes.append(arr.shape)
                nz = int((arr != 0).sum())
                nonzeros.append(nz)
                loaded_ok += 1
            except Exception as e:
                print(f"     ✗ Load failed: {matched}: {e}")

    if loaded_ok > 0:
        s = shapes[0]; nz_pct = np.mean(nonzeros) / (13*81*81) * 100
        nz_ok = nz_pct > 5.
        print(f"   {'✓' if nz_ok else '⚠'} {sid}: shape={s}  "
              f"nonzero={nz_pct:.1f}%  "
              f"{'OK' if nz_ok else 'WARNING: mostly zeros!'}")
        if not nz_ok: PASS = False
    else:
        print(f"   ✗ {sid}: không load được file nào")
        PASS = False

# ─── 4. Kiểm tra Env_Data ─────────────────────────────────────────────────────
print(f"\n4. Kiểm tra Env_Data (từng storm):")
env_dir = os.path.join(ROOT, "Env_Data")

for sid, year, name, raw, timestamps in storm_info:
    storm_dir_candidates = [
        os.path.join(env_dir, year, name),
        os.path.join(env_dir, year, name.lstrip('0') or '0'),
        os.path.join(env_dir, year, name.zfill(4)),
    ]
    storm_dir = None
    for sd in storm_dir_candidates:
        if os.path.isdir(sd):
            storm_dir = sd; break

    if storm_dir is None:
        print(f"   ✗ {sid}: Env_Data KHÔNG TÌM THẤY")
        PASS = False
        continue

    npy_files = sorted([f for f in os.listdir(storm_dir) if f.endswith('.npy')])
    if not npy_files:
        print(f"   ✗ {sid}: Env_Data rỗng")
        PASS = False
        continue

    # Load một file và kiểm tra keys
    fpath = os.path.join(storm_dir, npy_files[0])
    try:
        raw_env = np.load(fpath, allow_pickle=True)
        if raw_env.ndim == 0: raw_env = raw_env.item()
        if isinstance(raw_env, dict):
            keys = list(raw_env.keys())
            has_u500 = 'u500_mean' in raw_env
            has_v500 = 'v500_mean' in raw_env
            has_mv   = 'move_velocity' in raw_env
            print(f"   ✓ {sid}: dict {len(keys)} keys  "
                  f"u500={'✓' if has_u500 else '✗'}  "
                  f"v500={'✓' if has_v500 else '✗'}  "
                  f"move_velocity={'✓' if has_mv else '✗'}")
            # Check move_velocity range
            if has_mv:
                mv = float(raw_env['move_velocity'])
                norm_ok = 0. <= mv <= 1.
                if not norm_ok:
                    print(f"     ⚠ move_velocity={mv:.3f} out of [0,1]!")
        else:
            arr = np.array(raw_env).flatten()
            print(f"   ✓ {sid}: array shape={arr.shape}  mean={arr.mean():.3f}")
    except Exception as e:
        print(f"   ✗ {sid}: {e}")
        PASS = False

# ─── 5. Kiểm tra SVE computation ─────────────────────────────────────────────
print(f"\n5. Kiểm tra SVE từ Data3d thực tế:")
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from src_track.data.dataset import extract_sve_features, SVE_DIMS, decode_data1d
    
    for sid, year, name, raw, timestamps in storm_info[:2]:
        storm_dir_candidates = [
            os.path.join(data3d_dir, year, name),
            os.path.join(data3d_dir, year, name.lstrip('0') or '0'),
            os.path.join(data3d_dir, year, name.zfill(4)),
        ]
        storm_dir = None
        for sd in storm_dir_candidates:
            if os.path.isdir(sd): storm_dir = sd; break
        if not storm_dir: continue
        
        npy_files = sorted([f for f in os.listdir(storm_dir) if f.endswith('.npy')])
        if not npy_files: continue
        
        arr = np.load(os.path.join(storm_dir, npy_files[0])).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 13:
            arr = arr.transpose(2,0,1)
        
        sve = extract_sve_features(arr)
        has_nan = np.isnan(sve).any()
        has_inf = np.isinf(sve).any()
        nonzero = (sve != 0).sum()
        print(f"   {'✓' if not has_nan and not has_inf and nonzero > 5 else '⚠'} {sid}: "
              f"SVE shape={sve.shape}  nonzero={nonzero}/{SVE_DIMS}  "
              f"range=[{sve.min():.2f},{sve.max():.2f}]  "
              f"NaN={has_nan}  Inf={has_inf}")
        if has_nan or has_inf or nonzero < 5:
            print(f"     ⚠ SVE mostly zero/NaN — ERA5 data có vấn đề!")
            PASS = False
except ImportError:
    print("   [skip] src_track không available tại path này")

# ─── 6. Kiểm tra decode_data1d ────────────────────────────────────────────────
print(f"\n6. Kiểm tra decode_data1d scale:")
sid, year, name, raw, timestamps = storm_info[0]
phys = (raw * np.array([50,50,50,25]) + np.array([1800,0,960,40])) / np.array([10,10,1,1])
lon, lat = phys[:,0], phys[:,1]
pres, wnd = phys[:,2], phys[:,3]
lon_scs = (100 <= lon).all() and (lon <= 130).all()
lat_scs = (0 <= lat).all()  and (lat <= 30).all()
pres_ok = (800 <= pres).all() and (pres <= 1020).all()
wnd_ok  = (wnd >= 0).all()
print(f"   {'✓' if lon_scs else '✗'} lon: [{lon.min():.1f},{lon.max():.1f}]° (expect 100-130°)")
print(f"   {'✓' if lat_scs else '✗'} lat: [{lat.min():.1f},{lat.max():.1f}]° (expect 0-30°)")
print(f"   {'✓' if pres_ok else '✗'} pres: [{pres.min():.0f},{pres.max():.0f}] hPa (expect 800-1020)")
print(f"   {'✓' if wnd_ok else '✗'} wnd: [{wnd.min():.0f},{wnd.max():.0f}] kt")
if not (lon_scs and lat_scs and pres_ok and wnd_ok): PASS = False

# ─── Final ────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
if PASS:
    print("  ✅ ALL CHECKS PASSED — data OK, tiếp tục train")
else:
    print("  ❌ CÓ VẤN ĐỀ — cần debug trước khi train")
print(f"{'='*65}\n")

# subregion_selector.py — YAGO seeds → square (L∞) queries (100,200,1000,2000)
# Files expected next to this script:
#   - pid.txt                 (id lon lat OR lon lat OR lat lon — auto-detected)
#   - yago_popular.txt        (seeds: Node,Node id,lat,lon)  OR old bbox lines
# Output:
#   - yago_square/yago_<Node>_K{100|200|1000|2000}.pkl
from __future__ import annotations
import os, re, pickle
from typing import List, Tuple, Dict
import numpy as np
# ADD near the top with the other imports
from models import Place


HERE = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(HERE, "pid.txt")
POPULAR_FILE = os.path.join(HERE, "yago_popular.txt")
OUT_DIR = os.path.join(HERE, "yago_square")
K_TARGETS = [5000]

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# OLD: returns List[Tuple[int,float,float]]
# def read_pid_points(...)

# NEW: return List[Place] with coords = (lon, lat)
def read_pid_points(path: str) -> List[Place]:
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            nums = [float(x) for x in FLOAT_RE.findall(s.replace(",", " "))]
            if len(nums) >= 3:
                raw.append((nums[0], nums[1], nums[2]))     # maybe_id, a, b
            elif len(nums) == 2:
                raw.append((float("nan"), nums[0], nums[1]))# no id -> assign later

    if not raw:
        raise ValueError(f"No numeric rows in {path}")

    a = np.array([r[1] for r in raw], dtype=float)
    b = np.array([r[2] for r in raw], dtype=float)

    def score(lat, lon):
        valid = ((-90 <= lat) & (lat <= 90) & (-180 <= lon) & (lon <= 180)).sum()
        return valid + 0.001 * ((lon.max()-lon.min()) - (lat.max()-lat.min()))

    # detect orientation
    if score(a, b) >= score(b, a):
        lat, lon = a, b
    else:
        lat, lon = b, a

    ids = np.array([int(r[0]) if not np.isnan(r[0]) else -1 for r in raw], dtype=int)
    need_seq = ids < 0
    if need_seq.any():
        ids[need_seq] = np.arange(1, need_seq.sum() + 1, dtype=int)

    lon = np.clip(lon, -180, 180)
    lat = np.clip(lat, -90, 90)

    # RETURN Place objects with coords = (x=lon, y=lat)
    places: List[Place] = [Place(int(i), (float(lo), float(la))) for i, lo, la in zip(ids, lon, lat)]
    return places

def load_yago_seeds(path: str) -> List[Tuple[str, int, float, float]]:
    """
    Parse yago_popular.txt as a list of starting points:
      Preferred: Node,Node id,lat,lon
      Legacy box: Node,Node id,Total,Distinct,min_lat,min_lon,max_lat,max_lon  -> center used
    Returns list of (name, node_id, lat, lon).
    """
    seeds: List[Tuple[str, int, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.lower().startswith("node"):
                continue
            parts = [p for p in ln.split(",") if p != ""]
            nums = [float(x) for x in FLOAT_RE.findall(ln)]

            # Seed format: name,id,lat,lon -> expect at least 2 floats
            if len(parts) >= 4 and len(nums) >= 2 and len(nums) < 4:
                name = parts[0]
                node_id = int(float(parts[1]))
                lat, lon = float(parts[2]), float(parts[3])
                seeds.append((name, node_id, lat, lon))
                continue

            # Legacy bbox: name,id,total,distinct,min_lat,min_lon,max_lat,max_lon
            if len(parts) >= 8 and len(nums) >= 4:
                name = parts[0]
                node_id = int(float(parts[1]))
                min_lat, min_lon, max_lat, max_lon = [float(x) for x in nums[-4:]]
                lat = 0.5 * (min_lat + max_lat)
                lon = 0.5 * (min_lon + max_lon)
                seeds.append((name, node_id, lat, lon))
                continue

            # If we’re here, line is ambiguous — try last-resort: take last two numbers as lat/lon
            if len(nums) >= 2:
                name = parts[0]
                node_id = int(float(parts[1])) if len(parts) > 1 else 0
                lat, lon = float(nums[-2]), float(nums[-1])
                seeds.append((name, node_id, lat, lon))

    if not seeds:
        raise ValueError(f"No seeds parsed from {path}")
    return seeds

# OLD:
# def build_nested_square_queries(points: List[Tuple[int,float,float]], ...)-> Dict[int, List[Tuple...]]

# NEW: operate on Places and return Dict[int, List[Place]]
def build_nested_square_queries(places: List[Place],
                                center_lon: float, center_lat: float,
                                Ks: List[int]) -> Dict[int, List[Place]]:
    ids  = np.asarray([p.id for p in places], dtype=int)
    lons = np.asarray([p.coords[0] for p in places], dtype=float)  # x
    lats = np.asarray([p.coords[1] for p in places], dtype=float)  # y

    d_inf = np.maximum(np.abs(lons - center_lon), np.abs(lats - center_lat))
    order = np.lexsort((ids, d_inf))  # by L∞ then id for stability
    ranked = [places[i] for i in order]

    out: Dict[int, List[Place]] = {}
    for K in sorted(set(int(k) for k in Ks)):
        out[K] = ranked[:min(K, len(ranked))]
    return out

def save_queries(nested, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    prev = None
    clean = name
    if clean.startswith("yago_"):   # prevent double prefix
        clean = clean[len("yago_"):]
    for K in sorted(nested.keys()):
        cur_ids = {p.id for p in nested[K]}
        if prev is not None and not prev.issubset(cur_ids):
            raise AssertionError(f"[NEST] violation at K={K} for {name}")
        prev = cur_ids
        path = os.path.join(out_dir, f"yago_{_norm(clean)}_K{K}.pkl")
        with open(path, "wb") as f:
            pickle.dump(nested[K], f)  # <-- now lists of Place
        print(f"[OK] {path}  (# {len(nested[K])})")

def _norm(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

# ------------------- main ----------------------
def main():
    # 1) read the universe of YAGO places
    universe = read_pid_points(PID_FILE)  # [(id, lon, lat)]
    print(f"[INFO] Loaded {len(universe):,} points from pid.txt")

    # 2) read seeds (starting points)
    seeds = load_yago_seeds(POPULAR_FILE)  # [(name, node_id, lat, lon)]
    print(f"[INFO] Loaded {len(seeds)} seeds from yago_popular.txt")

    # 3) per seed → nearest (L∞) K places; save nested
    for name, node_id, lat, lon in seeds:
        nested = build_nested_square_queries(universe, center_lon=lon, center_lat=lat, Ks=K_TARGETS)
        save_queries(nested, OUT_DIR, name)

    print("[DONE] All queries built.")

if __name__ == "__main__":
    main()

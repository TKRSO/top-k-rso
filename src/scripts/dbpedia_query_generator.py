# subregion_selector.py
# -----------------------------------------------------------------------------
# Rebuild DBpedia region datasets using *square* (L∞) queries and a single,
# fixed center per region. For each region:
#   1) estimate a robust center,
#   2) build ONE global order by L∞ distance to that center (tie-break by id),
#   3) slice prefixes for each K in K_TARGETS -> PERFECT nesting across K.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import csv
import glob
import pickle
from typing import List, Tuple, Dict, Iterable, Optional
import numpy as np

# Your project models: Place must have .id and .coords -> (x, y)
from models import Place


# ======== CONFIG (adjust paths to your repo layout if needed) =================
DATA_ROOT   = "dbpedia"                      # Path to your input data
OUTPUT_DIR  = "dbpedia_output"        # A new folder for the output
REGIONS_CSV = None # e.g. "datasets/regions.csv" if you want a curated list

# K values to generate for *each* region/query
K_TARGETS: List[int] = [5000]


# ======== LOADING UTILITIES ====================================================
def load_places_from_pkl(path: str) -> List[Place]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Accept either a list[Place] or raw tuples (id, x, y)
    if data and not isinstance(data[0], Place):
        data = [Place(int(pid), (float(x), float(y))) for pid, x, y in data]
    return data


def load_places_from_txt(path: str) -> List[Place]:
    """
    TXT format: each line -> id x y
    """
    out: List[Place] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            pid, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            out.append(Place(pid, (x, y)))
    return out


def load_popular_regions(csv_path: str) -> List[str]:
    """
    Optional: CSV with a 'name' column; we only need region names here.
    If you keep bbox columns too, they’re ignored in this implementation.
    """
    names: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Node") or "").strip()
            if name:
                names.append(name)
    return names


def discover_region_names_from_data(data_root: str) -> List[str]:
    """
    Fallback: infer region names from files under DATA_ROOT.
    Accepts *.pkl or *.txt. e.g. datasets/dbpedia_raw/Athens.pkl -> 'Athens'
    """
    names = set()
    for pat in ("*.pkl", "*.txt"):
        for p in glob.glob(os.path.join(data_root, pat)):
            base = os.path.basename(p)
            stem = base.rsplit(".", 1)[0]
            names.add(stem)
    return sorted(names)


def load_region_places(name: str) -> List[Place]:
    """
    Try {DATA_ROOT}/{name}.pkl first, then {name}.txt.
    """
    pkl_path = os.path.join(DATA_ROOT, f"{name}.pkl")
    txt_path = os.path.join(DATA_ROOT, f"{name}.txt")
    if os.path.exists(pkl_path):
        return load_places_from_pkl(pkl_path)
    if os.path.exists(txt_path):
        return load_places_from_txt(txt_path)
    raise FileNotFoundError(f"No raw data for region '{name}' under {DATA_ROOT} (.pkl/.txt).")


# ======== CORE GEOMETRY / SELECTION ===========================================
def compute_fixed_center(places: List[Place], *, method: str = "median") -> np.ndarray:
    """
    Compute a robust center once per query/region.
    method: "median" (robust) or "mean" (classic).
    """
    if not places:
        raise ValueError("compute_fixed_center: empty places")
    pts = np.asarray([p.coords for p in places], dtype=float)
    if method == "median":
        cx = float(np.median(pts[:, 0]))
        cy = float(np.median(pts[:, 1]))
    elif method == "mean":
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
    else:
        raise ValueError(f"Unknown center method: {method}")
    return np.array([cx, cy], dtype=float)

def load_queries_with_bbox(csv_path: str) -> List[Dict]:
    """
    Loads unique query names and their bounding boxes from the popular regions file.
    Expects header: Node,...,PopularSubRegion,Distance,...
    The last 4 values are assumed to be min_lat, min_lon, max_lat, max_lon.
    """
    queries = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        for row in reader:
            if not row: continue
            name = row[0]
            if name not in queries:
                try:
                    # The last four comma-separated values are the bounding box
                    bbox = [float(c) for c in row[-4:]]
                    queries[name] = {
                        "name": name,
                        "bbox": {"min_lat": bbox[0], "min_lon": bbox[1], "max_lat": bbox[2], "max_lon": bbox[3]}
                    }
                except (ValueError, IndexError):
                    print(f"[WARN] Could not parse bounding box for row: {row}")
                    continue
    return list(queries.values())

def build_nested_square_subsets(
    places: List[Place],
    center: np.ndarray,
    K_values: Iterable[int],
) -> Dict[int, List[Place]]:
    """
    Build PERFECTLY NESTED subsets using L∞ distance to 'center'.
    One global order: (dist∞ asc, id asc)  -> prefixes for each K.
    Returns: {K: List[Place]}
    """
    Ks = sorted({int(k) for k in K_values})
    if not places:
        return {K: [] for K in Ks}

    pts   = np.asarray([p.coords for p in places], dtype=float)
    dists = np.max(np.abs(pts - center[None, :]), axis=1)

    try:
        ids = np.asarray([p.id for p in places])
    except Exception as e:
        raise AttributeError("Place must expose a stable 'id' for tie-breaking.") from e

    order = np.lexsort((ids, dists))           # primary: dists; secondary: ids
    ordered = [places[i] for i in order]       # ONE deterministic ranking

    out: Dict[int, List[Place]] = {}
    for K in Ks:
        K_eff = min(K, len(ordered))
        out[K] = ordered[:K_eff]
    return out


# ======== PERSISTENCE & HELPERS ===============================================
def _norm_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def save_nested_subsets(
    nested: Dict[int, List[Place]],
    *,
    out_dir: str,
    name: str,
    ensure_nested: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Optional safety: verify subset property across increasing K
    if ensure_nested:
        prev_ids: Optional[set] = None
        for K in sorted(nested.keys()):
            cur_ids = {p.id for p in nested[K]}
            if prev_ids is not None and not prev_ids.issubset(cur_ids):
                raise AssertionError(f"[NESTING VIOLATION] at K={K} for region '{name}'")
            prev_ids = cur_ids

    for K in sorted(nested.keys()):
        fname = f"dbpedia_{_norm_name(name)}_K{K}.pkl"
        path = os.path.join(out_dir, fname)
        with open(path, "wb") as f:
            pickle.dump(nested[K], f)
        print(f"[OK] Saved square selection: {path} (#{len(nested[K])})")


# ======== MAIN PIPELINE ========================================================
def generate_datasets(
    *,
    data_root: str = DATA_ROOT,
    out_dir: str = OUTPUT_DIR,
    k_targets: Iterable[int] = K_TARGETS,
    center_method: str = "median",
) -> None:
    """
    For each query in the popular regions file, filter pid.txt by its
    bounding box and generate nested subsets for that query.
    """
    # 1) Load the master list of all places ONCE
    master_places_path = os.path.join(data_root, "pid.txt")
    print(f"[INFO] Loading all places from master file: {master_places_path}")
    try:
        all_places = load_places_from_txt(master_places_path)
    except FileNotFoundError as e:
        print(f"[ERROR] Master pid.txt file not found. {e}")
        return
    print(f"[INFO] Loaded {len(all_places)} total places.")
    
    # Create a NumPy array for efficient filtering
    all_coords = np.asarray([p.coords for p in all_places], dtype=float)

    # 2) Load the 10 queries and their bounding boxes
    queries_path = os.path.join(data_root, "dbpedia_popular.txt")
    print(f"[INFO] Loading queries from: {queries_path}")
    queries = load_queries_with_bbox(queries_path)
    if not queries:
        print("[ERROR] No queries with valid bounding boxes found. Exiting.")
        return
    print(f"[INFO] Found {len(queries)} unique queries to process.")

    # 3) Process each query
    for query in queries:
        name = query["name"]
        bbox = query["bbox"]
        print(f"\n--- Processing query: {name} ---")

        # Filter places within the bounding box
        # Note: Latitude is Y (coords[1]), Longitude is X (coords[0])
        lat_mask = (all_coords[:, 0] >= bbox["min_lat"]) & (all_coords[:, 0] <= bbox["max_lat"])
        lon_mask = (all_coords[:, 1] >= bbox["min_lon"]) & (all_coords[:, 1] <= bbox["max_lon"])
        
        indices_in_box = np.where(lat_mask & lon_mask)[0]
        
        region_places = [all_places[i] for i in indices_in_box]

        if not region_places:
            print(f"[WARN] No places from pid.txt found within the bounding box for '{name}'. Skipping.")
            continue
        
        print(f"[INFO] Found {len(region_places)} places inside the bounding box.")

        # Run the standard analysis on this subset
        center = compute_fixed_center(region_places, method=center_method)
        nested = build_nested_square_subsets(region_places, center, k_targets)
        save_nested_subsets(nested, out_dir=out_dir, name=name, ensure_nested=True)

    print("\n[SUCCESS] All queries processed successfully.")
# ======== CLI =================================================================
def _parse_cli() -> Optional[dict]:
    import argparse
    ap = argparse.ArgumentParser(
        description="Rebuild *square* (L∞) DBpedia region datasets with PERFECT nesting across K."
    )
    ap.add_argument("--data-root", type=str, default=DATA_ROOT, help="Folder with raw region files (.pkl or .txt).")
    ap.add_argument("--out-dir", type=str, default=OUTPUT_DIR, help="Output folder to save S_K pickles.")
    ap.add_argument("--Ks", type=int, nargs="+", default=K_TARGETS, help="K values, e.g. 500 1000 2000.")
    ap.add_argument("--center-method", choices=["median", "mean"], default="median",
                    help="How to compute the fixed region center.")
    return vars(ap.parse_args())


# AFTER
def _main_cli():
    args = _parse_cli()
    generate_datasets(
        data_root=args["data_root"],
        out_dir=args["out_dir"],
        k_targets=args["Ks"],
        center_method=args["center_method"],
    )


if __name__ == "__main__":
    _main_cli()

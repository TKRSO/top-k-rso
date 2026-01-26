# dataset_store.py — K-only, uses your shapes ("flower","bubble","s_curve")

import os
import pickle
from typing import Iterable, Dict, List, Tuple, Optional
import numpy as np

from models import Place

# ---------- config hookup (square domain + defaults) ----------
try:
    import config as cfg
except Exception:  # allow quick runs without config present
    cfg = object()

SQ_X0: float = getattr(cfg, "SIM_SQUARE_X0", 0.0)
SQ_Y0: float = getattr(cfg, "SIM_SQUARE_Y0", 0.0)
SQ_SIDE: float = getattr(cfg, "SIM_SQUARE_SIDE", 100.0)

OUTPUT_DIR: str = getattr(cfg, "SIM_DATASETS_DIR", "datasets")
DEFAULT_K_VALUES: List[int] = list(getattr(cfg, "SIM_K_VALUES", [100, 200, 500, 1000, 2000]))
GLOBAL_SEED: Optional[int] = getattr(cfg, "SIM_SEED", None)

# ---------- rng helper ----------
def _rng(seed: Optional[int] = None) -> np.random.Generator:
    if seed is not None:
        return np.random.default_rng(seed)
    if GLOBAL_SEED is not None:
        return np.random.default_rng(GLOBAL_SEED)
    return np.random.default_rng()

# ---------- your generators, now K-only & square-based ----------

def generate_flower_shape(K: int) -> List[Place]:
    """
    Same 'flower' idea as before, but normalized into the square.
    Args: K only.
    """
    r = _rng()
    places: List[Place] = []

    n_core = K // 8
    n_petals = 5
    points_per_petal = max(1, (K - n_core) // n_petals)

    L = SQ_SIDE
    cx = SQ_X0 + L / 2.0
    cy = SQ_Y0 + L / 2.0
    spiral_radius = L * 0.35

    # core
    for _ in range(n_core):
        rr = r.uniform(0, spiral_radius * 0.1)
        th = r.uniform(0, 2 * np.pi)
        x = cx + rr * np.cos(th) + r.normal(0, 0.1)
        y = cy + rr * np.sin(th) + r.normal(0, 0.1)
        x = float(np.clip(x, SQ_X0, SQ_X0 + L))
        y = float(np.clip(y, SQ_Y0, SQ_Y0 + L))
        places.append(Place(len(places), (x, y)))

    # petals
    for p in range(n_petals):
        a0 = 2 * np.pi * p / n_petals
        for i in range(points_per_petal):
            t = (i / max(1, points_per_petal)) * np.pi
            rr = spiral_radius * (i / max(1, points_per_petal))
            x = cx + rr * np.cos(t + a0) + r.normal(0, 0.1)
            y = cy + rr * np.sin(t + a0) + r.normal(0, 0.1)
            x = float(np.clip(x, SQ_X0, SQ_X0 + L))
            y = float(np.clip(y, SQ_Y0, SQ_Y0 + L))
            places.append(Place(len(places), (x, y)))

    # pad if short
    while len(places) < K:
        x = float(np.clip(cx + r.normal(0, 0.2), SQ_X0, SQ_X0 + L))
        y = float(np.clip(cy + r.normal(0, 0.2), SQ_Y0, SQ_Y0 + L))
        places.append(Place(len(places), (x, y)))

    return places[:K]

def generate_bubble_clusters(K: int) -> List[Place]:
    """
    'bubble' = well-separated, *filled* circular clusters.
    - K-only (no k/G/range args)
    - centers laid out on a jittered lattice so clusters are far apart
    - points sampled mostly near the center (not a hollow ring)
    """
    r = _rng()
    L = SQ_SIDE

    # ---- how many bubbles from K (3..8) ----
    C = int(np.clip(round(np.sqrt(K / 300.0)) + 2, 3, 8))

    # ---- lattice of well-spaced centers + jitter ----
    g = int(np.ceil(np.sqrt(C)))                  # cells per axis
    step = L / g
    jitter_frac = 0.18                            # ± this * step
    # choose radius to avoid overlap even with jitter
    radius = step * 0.26                          # keep < 0.3 to stay safe with jitter
    margin = radius + jitter_frac * step + L * 0.02

    xs = np.linspace(SQ_X0 + margin, SQ_X0 + L - margin, g)
    ys = np.linspace(SQ_Y0 + margin, SQ_Y0 + L - margin, g)
    grid_centers = np.array([(x, y) for y in ys for x in xs], dtype=float)
    r.shuffle(grid_centers)                       # random pick of C cells
    centers = []
    for cx, cy in grid_centers[:C]:
        cx += (r.random() * 2 - 1) * jitter_frac * step
        cy += (r.random() * 2 - 1) * jitter_frac * step
        centers.append((float(cx), float(cy)))

    # ---- split K across bubbles (at least 1 each) ----
    w = r.dirichlet(np.ones(C))
    counts = np.maximum(1, np.floor(w * K).astype(int))
    while counts.sum() < K: counts[r.integers(0, C)] += 1
    while counts.sum() > K:
        j = r.integers(0, C)
        if counts[j] > 1: counts[j] -= 1

    # ---- sample inside discs (center-heavy) ----
    # interior mass dominates; a small fraction near the rim adds "bubbly" feel
    interior_frac = 0.88       # 88% interior points
    alpha = 1.6                # r = R * U**alpha  (alpha>0.5 => center-concentrated)
    rim_jitter = 0.04          # rim radial jitter (×R)

    places: List[Place] = []
    for (cx, cy), cnt in zip(centers, counts):
        if cnt <= 0:
            continue
        i_cnt = int(max(0, np.floor(interior_frac * cnt)))
        b_cnt = int(cnt - i_cnt)

        # interior (center-heavy disc)
        if i_cnt > 0:
            theta = r.uniform(0, 2*np.pi, size=i_cnt)
            rr = radius * (r.random(i_cnt) ** alpha)           # pushes mass to center
            xs = cx + rr * np.cos(theta)
            ys = cy + rr * np.sin(theta)
            xs = np.clip(xs, SQ_X0, SQ_X0 + L)
            ys = np.clip(ys, SQ_Y0, SQ_Y0 + L)
            for x, y in zip(xs, ys):
                places.append(Place(len(places), (float(x), float(y))))

        # thin rim (not a hollow ring; just a sprinkle near the edge)
        if b_cnt > 0:
            theta = r.uniform(0, 2*np.pi, size=b_cnt)
            rr = radius * (1.0 - rim_jitter + rim_jitter * 2 * r.random(b_cnt))
            xs = cx + rr * np.cos(theta)
            ys = cy + rr * np.sin(theta)
            xs = np.clip(xs, SQ_X0, SQ_X0 + L)
            ys = np.clip(ys, SQ_Y0, SQ_Y0 + L)
            for x, y in zip(xs, ys):
                places.append(Place(len(places), (float(x), float(y))))

    # safety top-up (should rarely trigger)
    while len(places) < K:
        cx, cy = centers[0]
        ang = r.uniform(0, 2*np.pi)
        rad = radius * (r.random() ** alpha)
        x = float(np.clip(cx + rad * np.cos(ang), SQ_X0, SQ_X0 + L))
        y = float(np.clip(cy + rad * np.sin(ang), SQ_Y0, SQ_Y0 + L))
        places.append(Place(len(places), (x, y)))

    return places[:K]

def generate_s_curve(K: int) -> List[Place]:
    """
    's_curve' as before, rescaled into the square.
    Args: K only.
    """
    r = _rng()
    raw = []
    for i in range(K):
        t = i / max(1, K) * 4 * np.pi
        x = np.sin(t) + r.normal(0, 0.1)
        y = np.sin(t) * np.cos(t) + r.normal(0, 0.1)
        raw.append((x, y))

    xs, ys = zip(*raw)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # scale into square
    sx = SQ_SIDE / (max_x - min_x + 1e-12)
    sy = SQ_SIDE / (max_y - min_y + 1e-12)

    places = []
    for i, (x, y) in enumerate(raw):
        nx = SQ_X0 + (x - min_x) * sx
        ny = SQ_Y0 + (y - min_y) * sy
        nx = float(np.clip(nx, SQ_X0, SQ_X0 + SQ_SIDE))
        ny = float(np.clip(ny, SQ_Y0, SQ_Y0 + SQ_SIDE))
        places.append(Place(i, (nx, ny)))
    return places

# ---------- keep YOUR mapping exactly ----------
SHAPES = {
    "flower": generate_flower_shape,
    "bubble": generate_bubble_clusters,
    "s_curve": generate_s_curve,
}

# ---------- K-only saver (what you said is good) ----------
def save_datasets(
    K_values: Optional[Iterable[int]] = None,
    shapes: Optional[Iterable[str]] = None,
    out_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[int, Dict[str, str]]:
    """
    K-only dataset writer. Filenames: {shape}_K{K}.pkl
    """
    if seed is not None:
        np.random.seed(seed)

    out_path = out_dir or OUTPUT_DIR
    os.makedirs(out_path, exist_ok=True)

    Ks = list(K_values) if K_values is not None else list(DEFAULT_K_VALUES)
    shape_list = list(shapes) if shapes is not None else list(SHAPES.keys())

    produced: Dict[int, Dict[str, str]] = {}
    for K in Ks:
        produced[K] = {}
        for name in shape_list:
            gen = SHAPES[name]
            data = gen(int(K))          # <-- ONLY K
            path = os.path.join(out_path, f"{name}_K{K}.pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f)
            produced[K][name] = path
            print(f"Saved: {path}")
    return produced

if __name__ == "__main__":
    save_datasets()

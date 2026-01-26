import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict


class Place:
    def __init__(self, id: int, coords: Tuple[float, float]):
        self.id = id
        self.coords = np.array(coords)
        self.rF = random.choice([0.4, 0.6, 0.8])
        #self.rF = 0.0
        self.cHPF = 0.0


class Cell:
    def __init__(self, cell_id: Tuple[int, int]):
        self.id = cell_id
        self.places: List[Place] = []
        self.center: np.ndarray = None
        self.score: float = None

    def add(self, p: Place):
        self.places.append(p)
        self.center = None  

    def compute_center(self) -> np.ndarray:
        if self.center is None:
            coords = np.array([p.coords for p in self.places])
            self.center = coords.mean(axis=0)
        return self.center

    def size(self) -> int:
        return len(self.places)

from typing import Dict, List, Tuple
import math

# Minimal interfaces assumed from your codebase:
# class Place:  # must have .coords -> (x, y)
# class Cell:   # must have .add(place) and .size()

class SquareGrid:
    """
    Grid that tightly wraps S's bounding *rectangle* (not forced square).
    It chooses Ax × Ay = G using divisors of G so the grid aspect ratio
    follows S's aspect ratio as closely as possible.

    - Bounds: tight rectangle around S (no padding).
    - Ax × Ay = G exactly (uses nearest divisor to target Ax).
    - Cells are rectangles (square only if S is ~square or G allows Ax==Ay).
    - Stores only non-empty cells by default (virtual grid).
    """

    def __init__(self, places: List["Place"], G: int, precreate: bool = False):
        if not places:
            raise ValueError("SquereGrid requires non-empty 'places'.")
        if not isinstance(G, int) or G <= 0:
            raise ValueError("G must be a positive integer.")

        self.places = places
        self.G = G

        # ---- tight rectangular bounds from S ----
        xs = [p.coords[0] for p in places]
        ys = [p.coords[1] for p in places]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        width  = max(self.x_max - self.x_min, 1e-12)
        height = max(self.y_max - self.y_min, 1e-12)
        aspect = width / height  # >1 = wider than tall

        # ---- choose Ax × Ay = G to match aspect ratio ----
        Ax, Ay = self._best_axes(G, aspect)
        self.Ax, self.Ay = Ax, Ay  # cells per axis (x, y)

        # ---- cell sizes (epsilon keeps max-edge inside last cell) ----
        eps = 1e-12
        self.cell_w = (width  - eps) / self.Ax
        self.cell_h = (height - eps) / self.Ay

        # ---- grid storage ----
        self._grid: Dict[Tuple[int, int], "Cell"] = {}
        if precreate:
            for gx in range(self.Ax):
                for gy in range(self.Ay):
                    self._grid[(gx, gy)] = Cell((gx, gy))

        self._assign_to_cells()

    # ---------- internals ----------
    @staticmethod
    def _divisors(n: int) -> List[int]:
        """All positive divisors of n."""
        ds = set()
        r = int(math.isqrt(n))
        for a in range(1, r + 1):
            if n % a == 0:
                ds.add(a)
                ds.add(n // a)
        return sorted(ds)

    def _best_axes(self, G: int, aspect: float) -> Tuple[int, int]:
        """
        Pick Ax, Ay (integers, Ax*Ay==G) so Ax/Ay ≈ aspect and
        Ax is as close as possible to sqrt(G*aspect).
        """
        target_Ax = math.sqrt(G * max(aspect, 1e-12))
        best_Ax = 1
        best_err = float("inf")
        for a in self._divisors(G):
            err = abs(a - target_Ax)
            if err < best_err:
                best_err = err
                best_Ax = a
        Ax = best_Ax
        Ay = G // Ax
        return Ax, Ay

    def _to_index(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.x_min) / self.cell_w)
        gy = int((y - self.y_min) / self.cell_h)
        # clamp to valid range
        if gx < 0: gx = 0
        elif gx >= self.Ax: gx = self.Ax - 1
        if gy < 0: gy = 0
        elif gy >= self.Ay: gy = self.Ay - 1
        return gx, gy

    def _assign_to_cells(self) -> None:
        for p in self.places:
            key = self._to_index(*p.coords)
            if key not in self._grid:
                self._grid[key] = Cell(key)
            self._grid[key].add(p)

    # ---------- API ----------
    def get_grid(self) -> Dict[Tuple[int, int], "Cell"]:
        """Dict of cells (keys=(gx,gy)); only non-empty unless precreate=True."""
        return self._grid

    def get_full_cells(self) -> List["Cell"]:
        """List of non-empty cells (CL)."""
        return [c for c in self._grid.values() if c.size() > 0]

    def get_all_cells(self) -> List["Cell"]:
        """Return a list of ALL cells (size = G), including empty ones."""
        all_cells = []
        for gx in range(self.Ax):
            for gy in range(self.Ay):
                key = (gx, gy)
                if key not in self._grid:
                    self._grid[key] = Cell(key)   # create empty cell
                all_cells.append(self._grid[key])
        return all_cells

    
    def total_cells(self) -> int:
        """Total cells = G (Ax × Ay)."""
        return self.Ax * self.Ay

    def dims(self) -> Tuple[int, int]:
        """(Ax, Ay) cells per axis."""
        return self.Ax, self.Ay

    def stats(self) -> str:
        non_empty = len(self.get_full_cells())
        K = len(self.places)
        avg_occ = K / max(1, non_empty)
        return (f"Ax×Ay={self.Ax}×{self.Ay} total={self.total_cells()} "
                f"non_empty={non_empty} K={K} avg_occ={avg_occ:.2f} "
                f"span=[{self.x_min:.6g},{self.x_max:.6g}]×[{self.y_min:.6g},{self.y_max:.6g}]")

# --- FullGrid: same geometry as SquareGrid, but fully materialized (empties included) ---
class FullGrid(SquareGrid):
    """
    Full materialized grid: pre-creates all Ax×Ay = G cells so FL == G always.
    Use when you need the FULL list of cells (including empty ones).
    """
    def __init__(self, places: List["Place"], G: int, pad_ratio: float = 0.0):
        # keep SquareGrid's geometry/assignments, but force full pre-create
        super().__init__(places, G, precreate=True)
        # Ensure dictionary contains every cell (already true with precreate=True)
        for gx in range(self.Ax):
            for gy in range(self.Ay):
                self._grid.setdefault((gx, gy), Cell((gx, gy)))

    # Full list (FL): ALL cells including empties
    def get_all_cells(self) -> List["Cell"]:
        return [self._grid[(gx, gy)] for gx in range(self.Ax) for gy in range(self.Ay)]

    # Optional helper: assign geometric centers to empty cells to avoid mean-of-empty warnings
    def ensure_empty_cell_centers(self) -> None:
        for gx in range(self.Ax):
            for gy in range(self.Ay):
                c = self._grid[(gx, gy)]
                if c.center is None and c.size() == 0:
                    cx = self.x_min + (gx + 0.5) * self.cell_w
                    cy = self.y_min + (gy + 0.5) * self.cell_h
                    c.center = np.array([cx, cy])

class QuadNode:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, depth: int):
        self.bounds = (x_min, y_min, x_max, y_max)
        self.depth = depth
        self.children: List['QuadNode'] = []
        self.places: List = [] 
        self.is_leaf = True

    def insert(self, place, m: int, max_d: int):
        if not self.is_leaf:
            self._get_child(place).insert(place, m, max_d)
            return

        self.places.append(place)

        if len(self.places) > m and self.depth < max_d:
            self.split(m, max_d)

    def split(self, m, max_d):
        self.is_leaf = False
        x_min, y_min, x_max, y_max = self.bounds
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        next_depth = self.depth + 1

        self.children = [
            QuadNode(x_min, mid_y, mid_x, y_max, next_depth), # NW
            QuadNode(mid_x, mid_y, x_max, y_max, next_depth), # NE
            QuadNode(x_min, y_min, mid_x, mid_y, next_depth), # SW
            QuadNode(mid_x, y_min, x_max, mid_y, next_depth)  # SE
        ]

        existing_places = self.places
        self.places = []
        
        for p in existing_places:
            self._get_child(p).insert(p, m, max_d)

    def _get_child(self, place) -> 'QuadNode':
        # FIX: Access coordinates via .coords tuple
        # Assumes place.coords = (x, y)
        x, y = place.coords[0], place.coords[1]
        
        x_min, y_min, x_max, y_max = self.bounds
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2

        if x < mid_x:
            if y >= mid_y: return self.children[0] # NW
            else:          return self.children[2] # SW
        else:
            if y >= mid_y: return self.children[1] # NE
            else:          return self.children[3] # SE

    def get_all_leaves(self) -> List['QuadNode']:
        if self.is_leaf:
            return [self] if len(self.places) > 0 else []
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves

class QuadTree:
    def __init__(self, S: List, m: int, max_d: int):
        if not S:
            self.root = None
            return
            
        # FIX: Update bounds calculation to use .coords
        min_x = min(p.coords[0] for p in S)
        max_x = max(p.coords[0] for p in S) + 0.00001
        min_y = min(p.coords[1] for p in S)
        max_y = max(p.coords[1] for p in S) + 0.00001
        
        self.root = QuadNode(min_x, min_y, max_x, max_y, 0)
        
        for p in S:
            self.root.insert(p, m, max_d)
            
    def get_leaves(self):
        if not self.root: return []
        return self.root.get_all_leaves()
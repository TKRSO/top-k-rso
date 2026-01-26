import time
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from alg.HPF_eq import HPF, HPFR
from alg.baseline_iadu import base_precompute, baseline_iadu_algorithm
from models import Place, Cell, SquareGrid, FullGrid
import config as cfg
# Global Grid Config

#################################################################################################################################################################################
import heapq

class MinHeap:
    def __init__(self, k: int):
        self.k = k
        self.heap = []  # stores (cHPF, id, Place)

    def push(self, place: Place):
        entry = (-place.cHPF, place.id, place)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        else:
            if entry[0] < self.heap[0][0]:
                heapq.heappushpop(self.heap, entry)

    def pop(self):
        # >>> empty-safe: return None instead of raising
        if not self.heap:
            return None
        return heapq.heappop(self.heap)[2]

    def peek(self):
        # >>> empty-safe
        return self.heap[0][2] if self.heap else None

    def is_empty(self):
        return not self.heap

    def __len__(self):
        return len(self.heap)

##########################################################################################################################################
###########################################################################################################################################################
def grid_based_iadu_algorithm(S: list[Place], CL: list[Cell], W: float, psS ,sS: dict, k: int) -> Tuple[list[Place], float]:

    K = len(S)
    # Create heap per cell with top-k places by initial cHPF = rF + pSS
    TkH = {}
    heads = {}
    R = []
            
    heap_time_start = time.time()
    for cell in CL:
        TkH[cell.id] = MinHeap(k)
        for place in cell.places:
            #place.cHPF = psS[place.id] + place.rF
            #place.cHPF = psS[place.id]
            place.cHPF = 0
            TkH[cell.id].push(place)
        heads[cell.id] = TkH[cell.id].pop()
    
    while len(R) != k:
        # Get the top place across all heap heads
        max_score = 0.0
        curID = None
        
        for id, head in heads.items():
            if head.cHPF >= max_score:
                curMP = head
                curH = TkH[id]
                curID = id
                max_score = curMP.cHPF
        
        if curID is None:
            break
    
        # Put next head in array
        del heads[curID]
        if not curH.is_empty():
            heads[curID] = curH.pop()
        
        # Update of new head of curH
        if not curH.is_empty():
            if len(R) < k:
                for p in R:
                    heads[curID].cHPF += (K - k) * (heads[curID].rF - p.rF) / (k - 1) + (psS[heads[curID].id] + psS[p.id]) / (k - 1) - 2 * W * sS[(heads[curID].id, p.id)]



        # Add curMP to R
        R.append(curMP)

        # Update each heap's head with contribution from curMP
        if len(R) < k:
            for head in heads.values():
                head.cHPF += (K - k) * (head.rF - curMP.rF) / (k - 1) + (psS[head.id] + psS[curMP.id]) / (k - 1) - 2 * W * sS[(head.id, curMP.id)]
    selection_time = time.time() - heap_time_start
    return R, selection_time

def old_grid_iadu_algorithm(S: list[Place], CL: list[Cell], W: float, psS ,sS: dict, k: int) -> Tuple[list[Place], float]:

    K = len(S)
    # Create heap per cell with top-k places by initial cHPF = rF + pSS
    TkH = {}
    heads = {}
    R = []
    
    heap_time_start = time.time()
    # --- build heaps / heads ---
    for cell in CL:
        TkH[cell.id] = MinHeap(k)
        for place in cell.places:
            place.cHPF = psS[place.id] + place.rF
            TkH[cell.id].push(place)
        mp = TkH[cell.id].pop()          # <-- may be None if cell had <1 valid push
        heads[cell.id] = mp

    # --- selection loop ---
    while len(R) != k:
        max_score = float("-inf")
        curID = None
        for id, head in heads.items():
            if head is None:    # <<< safety: skip empty heaps
                continue
            if head.cHPF >= max_score:
                curMP = head
                curH = TkH[id]
                curID = id
                max_score = curMP.cHPF

        if curID is None:                # <<< safety: nothing to pick
            break

        # Put next head in array
        del heads[curID]
        nxt = curH.pop()                 # <<< may be None now
        if nxt is not None:
            heads[curID] = nxt

        # Update of new head of curH
        if not curH.is_empty():
            if len(R) < k:
                for p in R:
                    heads[curID].cHPF += (K - k) * (heads[curID].rF - p.rF) / (k - 1) + (psS[heads[curID].id] + psS[p.id]) / (k - 1) - 2 * W * sS[(heads[curID].id, p.id)]

        # Add curMP to R
        R.append(curMP)

        # Update each heap's head with contribution from curMP
        if len(R) < k:
            for head in heads.values():
                if head is not None:
                    head.cHPF += (K - k) * (head.rF - curMP.rF) / (k - 1) + (psS[head.id] + psS[curMP.id]) / (k - 1) - 2 * W * sS[(head.id, curMP.id)]
    selection_time = time.time() - heap_time_start
    return R, selection_time


###############################################################################################################################################################################
###############################################################################################################################################################################
def virtual_grid_based_algorithm(CL: List[Cell], S: List[Place]) -> Tuple[ Dict[int, float], float]:

    cell_sS = {}
    pr = {cell.id: 0.0 for cell in CL}
    prep_time = 0.0
    centers = {cell.id: cell.compute_center() for cell in CL}
    cl_centers = list(centers.keys())
    
    # Compute max distance for normalization
    maxD = maxDistance(S)
    
    
    for ci in cl_centers:
        for cj in cl_centers:
            if ci == cj:
                cell_sS[(ci, cj)] = 1.0  # ensure diagonal exists
            else:
                d = np.linalg.norm(centers[ci] - centers[cj])
                cell_sS[(ci, cj)] = 1.0 - (d / maxD)


    prep_time_start = time.time()
    for i in range(len(CL)):
        ci = CL[i]
        for j in range(i , len(CL)):
            cj = CL[j]
            sim = 1
            if ci != cj:
                sim = cell_sS[(ci.id,cj.id)]
                
            # Update pr(ci) and pr(cj)
            pr[ci.id] += cj.size() * sim
            if ci != cj:
                pr[cj.id] += ci.size() * sim
        pr[ci.id] -= 1.0
        
        
    prep_time = time.time() - prep_time_start
    
    place_to_cell = map_place_to_cell(CL)
    
    psS = {p.id: 0.0 for p in S}
    sS = {}
    for pi in S:
        psS[pi.id] = pr[place_to_cell[pi.id]]
        for pj in S:
            if (place_to_cell[pi.id] != place_to_cell[pj.id]):
                sS[(pi.id, pj.id)] = cell_sS[(place_to_cell[pi.id], place_to_cell[pj.id])]
            else:
                if (pi.id != pj.id):
                    sS[(pi.id, pj.id)] = 1.0
                    
    return psS, sS, prep_time

def old_grid_precompute(CL: List[Cell], S: List[Place]) -> Tuple[ Dict[int, float], float]:

    cell_sS = {}
    pr = {cell.id: 0.0 for cell in CL}
    prep_time = 0.0
    centers = {cell.id: cell.compute_center() for cell in CL}
    cl_centers = list(centers.keys())
    
    # Compute max distance for normalization
    maxD = maxDistance(S)
    
    
    for ci in cl_centers:
        for cj in cl_centers:
            if ci == cj:
                cell_sS[(ci, cj)] = 1.0  # ensure diagonal exists
            else:
                d = np.linalg.norm(centers[ci] - centers[cj])
                cell_sS[(ci, cj)] = 1.0 - (d / maxD)


    prep_time_start = time.time()
    for i in range(len(CL)):
        ci = CL[i]
        for j in range(i , len(CL)):
            cj = CL[j]
            sim = 1
            if ci != cj:
                sim = cell_sS[(ci.id,cj.id)]
                
            # Update pr(ci) and pr(cj)
            pr[ci.id] += cj.size() * sim
            if ci != cj:
                pr[cj.id] += ci.size() * sim
        pr[ci.id] -= 1.0
        
        
    prep_time = time.time() - prep_time_start
    
    place_to_cell = map_place_to_cell(CL)
    
    psS = {p.id: 0.0 for p in S}
    sS = {}
    for pi in S:
        psS[pi.id] = pr[place_to_cell[pi.id]]
        for pj in S:
            if (place_to_cell[pi.id] != place_to_cell[pj.id]):
                sS[(pi.id, pj.id)] = cell_sS[(place_to_cell[pi.id], place_to_cell[pj.id])]
            else:
                if (pi.id != pj.id):
                    sS[(pi.id, pj.id)] = 1.0
                    
    return psS, sS, prep_time

def maxDistance(S: List[Place]) -> float:
    maxD = 0
    coords = np.array([p.coords for p in S])
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = np.linalg.norm(coords[i] - coords[j])
            maxD = max(maxD, d)
    return maxD

import pickle

def load_dataset(shape_name: str, K: int, k: int, G: int):
    path = f"datasets/{shape_name}_K{K}_k{k}_G{G}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


all_K_k = cfg.COMBO
from typing import Dict, List

def map_place_to_cell(CL: List['Cell']) -> Dict[int, Tuple[int, int]]:
    place_to_cell = {}
    for cell in CL:
        for p in cell.places:
            place_to_cell[p.id] = cell.id
    return place_to_cell

def plot_on_ax(ax, S, grid, cell_size, grid_bounds, title="", R=None):
    x_min, x_max, y_min, y_max = grid_bounds
    CL = list(grid.values())
    x_ids = [cid[0] for cid in grid]
    y_ids = [cid[1] for cid in grid]
    gx_min, gx_max = min(x_ids), max(x_ids)
    gy_min, gy_max = min(y_ids), max(y_ids)

    # Plot all places
    xs = [p.coords[0] for p in S]
    ys = [p.coords[1] for p in S]
    ax.scatter(xs, ys, color='lightblue', s=15, label='Places', zorder=2)

    # Plot selected R in red
    if R:
        rx = [p.coords[0] for p in R]
        ry = [p.coords[1] for p in R]
        ax.scatter(rx, ry, color='red', s=30, label='Selected R', zorder=3)

    # Grid rectangles
    for gx in range(gx_min, gx_max + 1):
        for gy in range(gy_min, gy_max + 1):
            x0 = x_min + gx * cell_size
            y0 = y_min + gy * cell_size
            cid = (gx, gy)
            face = 'white' if cid in grid and grid[cid].size() > 0 else 'lightgrey'
            rect = plt.Rectangle((x0, y0), cell_size, cell_size,
                                facecolor=face, edgecolor='black', linewidth=0.8, zorder=1)
            ax.add_patch(rect)

    for cell in CL:
        if cell.size() > 0:
            gx, gy = cell.id
            x0 = x_min + gx * cell_size
            y0 = y_min + gy * cell_size
            ax.text(x0 + 0.1, y0 + 0.1, f"{cell.size()}", fontsize=6, zorder=4)

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_aspect('equal')
    ax.set_title(title.capitalize())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

def base_iadu_on_grid(S: List[Place], k: int, W, G, exact_psS, exact_sS) -> Tuple[List[Place], Dict[int, float], Dict[int, float], float, float, float]:
    K = len(S)
    
    # Preparation step
    grid = SquareGrid(S, G)
    CL = list(grid.get_grid().values())
    psS, sS, prep_time = virtual_grid_based_algorithm(CL,S)
        
    # Run baseline IAdU algorithm
    R, selection_time = baseline_iadu_algorithm(S, K, k, W, psS, sS)
    
    # Compute final scores
    score, sum_psS, sum_psR, sum_rF = HPFR(R, exact_psS, exact_sS, W, len(S))
    
    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time


def grid_iadu(S: List[Place], k: int, W, G: int, optimal_psS, optimal_sS) -> Tuple[List[Place], Dict[int, float], Dict[int, float], float, float, float]:

    # Preparation step
    grid = SquareGrid(S, G)
    CL = grid.get_full_cells()
    psS, sS , prep_time = virtual_grid_based_algorithm(CL,S)
    # optimal_psS, optimal_sS, optimal_prep_time = base_precompute(S)

        
    # Run grid IAdU algorithm
    R, selection_time = grid_based_iadu_algorithm(S, CL, W, psS, sS, k)
    
    # Compute final scores
    score, sum_psS, sum_psR, sum_rF = HPFR(R, optimal_psS, optimal_sS, W, len(S))
    
    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time, len(CL)

def old_grid_iadu(S: List[Place], k: int, W, G: int, optimal_psS, optimal_sS) -> Tuple[List[Place], Dict[int, float], Dict[int, float], float, float, float]:

    # Preparation step
    grid = FullGrid(S, G)
    FL = grid.get_all_cells()
    grid.ensure_empty_cell_centers()
    psS, sS , prep_time = old_grid_precompute(FL,S)
    
    # Run grid IAdU algorithm
    R, selection_time = old_grid_iadu_algorithm(S, FL, W, psS, sS, k)
    
    # Compute final scores
    score, sum_psS, sum_psR, sum_rF = HPFR(R, optimal_psS, optimal_sS, W, len(S))

    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time, len(FL)
import copy
from typing import List, Tuple, Dict
import time
import numpy as np
from alg.HPF_eq import HPFR, HPFR_no_r
from models import Place, SquareGrid
import config as cfg

def baseline_iadu_algorithm(S: List[Place], K_full: int, k: int, W: float, psS, sS) -> Tuple[List[Place], float]:
    R = []
    K = K_full
    candidates = copy.deepcopy(S)

    for p in candidates:
        p.cHPF = psS[p.id] + (K - 1)* p.rF
        # p.cHPF = 0
    
    select_start = time.time()
    while len(R) < k:
        curMP = max(candidates, key=lambda p: p.cHPF)
        candidates.remove(curMP)
        R.append(curMP)
        if len(R) < k:
            for p in candidates:
                if p.id != curMP.id:
                    p.cHPF += (K - k) * (p.rF + curMP.rF) * cfg.wrf / (k - 1) + (psS[p.id] + psS[curMP.id]) / (k - 1) - 2 * W * spacial_proximity(sS, p, curMP)
    
    select_end = time.time()
    
    select_end - select_start
    return R, select_end - select_start

def baseline_iadu_algorithm_no_r(S: List[Place], K_full: int, k: int, W: float, psS, sS) -> Tuple[List[Place], float]:
    R = []
    K = K_full
    candidates = copy.deepcopy(S)

    for p in candidates:
        p.cHPF = 0
    
    select_start = time.time()
    while len(R) < k:
        curMP = max(candidates, key=lambda p: p.cHPF)
        candidates.remove(curMP)
        R.append(curMP)
        if len(R) < k:
            for p in candidates:
                if p.id != curMP.id:
                    p.cHPF += (psS[p.id] + psS[curMP.id]) / (k - 1) - 2 * W * spacial_proximity(sS, p, curMP)
    select_end = time.time()
    
    select_end - select_start
    return R, select_end - select_start

############################################################################################################
# use symmetric sS
def spacial_proximity(sS, pi, pj):
    return sS.get((pi.id, pj.id)) or sS.get((pj.id, pi.id)) or 0.0


def base_precompute(S: List[Place]) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    sS = {}
    maxD = maxDistance(S)
    # Initialize psS
    psS = {p.id: 0.0 for p in S}
    
    prep_start = time.time()
    for i in range(len(S)):
        pi = S[i]
        for j in range(i + 1, len(S)):
            pj = S[j]
            sim = 0
            if pi.id != pj.id:
                # Calculate Euclidean distance and similarity
                d = np.linalg.norm(pi.coords - pj.coords)
                sim = 1 - d / maxD
                
                # Store similarity in the dictionary
                sS[pi.id, pj.id] = sim
                # Update psS
                psS[pj.id] += sim
                psS[pi.id] += sim
    prep_end = time.time()
            
    return psS, sS, prep_end - prep_start

####################################################################################################
#####################################################################################################
# --- IAdU method ---
def iadu(S: List[Place], k: int, W, exact_psS, exact_sS, prep_time) -> Tuple[List[Place], Dict[int, float], Dict[int, float], float, float, float]:
    K = len(S)
        
    # Run baseline IAdU algorithm
    R, selection_time = baseline_iadu_algorithm(S, K, k, W, exact_psS, exact_sS)
    
    # Compute final scores
    score, sum_psS, sum_psR, sum_rF = HPFR(R, exact_psS, exact_sS, W, len(S))
    
    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time

# --- IAdU method ---
def iadu_no_r(S: List[Place], k: int, W) -> Tuple[List[Place], Dict[int, float], Dict[int, float], float, float, float]:
    K = len(S)
    # Preparation step
    exact_psS, exact_sS, prep_time = base_precompute(S)
        
    # Run baseline IAdU algorithm
    R, selection_time = baseline_iadu_algorithm_no_r(S, K, k, W, exact_psS, exact_sS)
    
    # Compute final scores
    score, sum_psS, sum_psR, sum_rF = HPFR_no_r(R, exact_psS, exact_sS, W, len(S))
    
    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time

# import pickle

# def load_dataset(name: str, K: int):
#     path = f"datasets/{name}_K{K}.pkl"
#     with open(path, "rb") as f:
#         return pickle.load(f)

def load_db_dataset(region_name: str, K: int) -> List[Place]:
    """
    Load a DBpedia subregion dataset (exact K places) from db_datasets folder.
    """
    import pickle, os
    from models import Place
    fname = f"dbpedia_{region_name}_K{K}.pkl"
    path = os.path.join("db_datasets", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dataset file found: {path}")
    with open(path, "rb") as f:
        data: List[Place] = pickle.load(f)
    return data

def load_yago_dataset(region_name: str, K: int) -> List[Place]:
    """
    Load a DBpedia subregion dataset (exact K places) from db_datasets folder.
    """
    import pickle, os
    from models import Place
    fname = f"yago_{region_name}_K{K}.pkl"
    path = os.path.join("yago_datasets", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dataset file found: {path}")
    with open(path, "rb") as f:
        data: List[Place] = pickle.load(f)
    return data
from typing import List
import os, pickle, re
from models import Place

def load_dataset(dataset_name: str, K: int) -> List[Place]:
    """
    Loads a pickled dataset based on its name and K value.
    This new version is flexible and works for "shapes" (bubble, flower)
    and "real" (dbpedia, yago) datasets, as they all
    follow the same "{dataset_name}_K{K}.pkl" format.
    """
    # Construct the path relative to this file (src/baseline_iadu.py)
    # Go up one level (to src/) and then into "datasets"
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
    file_path = os.path.join(base_path, f"{dataset_name}_K{K}.pkl")

    if not os.path.exists(file_path):
        print(f"Warning: Dataset file not found at {file_path}")
        print(f"       (Trying to load: dataset_name='{dataset_name}', K={K})")
        return []
    
    try:
        with open(file_path, "rb") as f:
            S = pickle.load(f)
        
        # Verify the loaded data is a list of Place objects
        if isinstance(S, list) and (len(S) == 0 or isinstance(S[0], Place)):
            return S
        else:
            print(f"Error: File {file_path} loaded, but did not contain a list of Place objects.")
            return []
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return []

def maxDistance(S: List[Place]) -> float:
    maxD = 0
    coords = np.array([p.coords for p in S])
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = np.linalg.norm(coords[i] - coords[j])
            maxD = max(maxD, d)
    return maxD

def plot_selected(S: List[Place], R: List[Place], title: str, ax, grid: SquareGrid = None, cell_stats: Dict = None):
    
    R_ids = {p.id for p in R}

    S_minus_R_coords = []
    R_coords = []

    for p in S:
        if p.id in R_ids:
            R_coords.append(p.coords)
        else:
            S_minus_R_coords.append(p.coords)

    if S_minus_R_coords:
        coords_S_minus_R = np.array(S_minus_R_coords)
        ax.scatter(coords_S_minus_R[:, 0], coords_S_minus_R[:, 1], c="lightblue", s=10, label="S - R")
    
    if R_coords:
        coords_R = np.array(R_coords)
        ax.scatter(coords_R[:, 0], coords_R[:, 1], c="red", s=25, label="R")
        
    ax.set_title(title)
    ax.legend()
    
    if grid:
        x_min, x_max = grid.x_min, grid.x_max
        y_min, y_max = grid.y_min, grid.y_max
        cell_w, cell_h = grid.cell_w, grid.cell_h
        Ax, Ay = grid.dims()

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        v_lines = [x_min + (i * cell_w) for i in range(1, Ax)]
        if v_lines:
            ax.vlines(v_lines, ymin=y_min, ymax=y_max, color='gray', linestyle='--', linewidth=0.5)
        
        h_lines = [y_min + (i * cell_h) for i in range(1, Ay)]
        if h_lines:
            ax.hlines(h_lines, xmin=x_min, xmax=x_max, color='gray', linestyle='--', linewidth=0.5)
            
        # --- NEW: Text annotation logic ---
        # Only runs if 'cell_stats' is provided
        if cell_stats:
            for cell_id, (total_count, selected_count) in cell_stats.items():
                if total_count > 0: # Only label cells that have points
                    gx, gy = cell_id
                    # Calculate center of the cell
                    cx = x_min + (gx + 0.5) * cell_w
                    cy = y_min + (gy + 0.5) * cell_h
                    
                    # Create the text
                    text = f"{selected_count}/{total_count}"
                    
                    # Add text to the plot
                    ax.text(cx, cy, text, 
                            ha='center', va='center', 
                            fontsize=8, color='black',
                            bbox=dict(facecolor='white', alpha=0.5, pad=0.1, boxstyle='round,pad=0.2'))
        # --- End of new logic ---
        
baseline_scores = []
baseline_prep_times = []
iadu_scores = []

all_K_k = cfg.COMBO



import sys
import os
import time
import math
import random
from typing import List, Dict, Tuple

# Ensure parent directory is in path to import models, HPF_eq, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import Place
from models import SquareGrid, QuadTree
from alg.HPF_eq import HPFR, HPFR_no_r

def grid_sampling(S: List[Place], k: int, W: float, G: int, optimal_psS, optimal_sS):

    t_prep_start = time.time()
    grid = SquareGrid(S, G)
    CL = grid.get_full_cells() # Get non-empty cells
    prep_time = time.time() - t_prep_start
        
    K = len(S)
    if K == 0 or k == 0:
        return [], 0.0, 0.0, 0.0, prep_time, 0.0, 0, {}
    
    R: List[Place] = []
    k_alloc: Dict[Tuple[int, int], int] = {} 
    cell_stats: Dict[Tuple[int, int], Tuple[int, int]] = {}
    remainders = [] 
    total_k_allocated = 0
    cell_map = {c.id: c for c in CL} 

    t_selection_start = time.time()
    
    for c in CL:
        if c.size() == 0:
            continue
        
        ideal = k * (c.size() / K)
        integer_part = math.floor(ideal)
        
        k_alloc[c.id] = integer_part
        total_k_allocated += integer_part
        remainders.append((c.id, ideal - integer_part))

    k_remaining = k - total_k_allocated
    
    remainders.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(min(k_remaining, len(remainders))):
        c_id_to_add = remainders[i][0]
        
        k_alloc[c_id_to_add] += 1
    
    for cell_id, num_to_pick in k_alloc.items():
        if num_to_pick > 0:
            cell = cell_map[cell_id]
            actual_pick = min(num_to_pick, cell.size())
            
            if actual_pick > 0:
                R.extend(random.sample(cell.places, actual_pick))

    selection_time = time.time() - t_selection_start
    
    
    for c in CL: # Use all non-empty cells
        cell_id = c.id
        total_count = c.size()
        selected_count = k_alloc.get(cell_id, 0) # Get from k_alloc
        cell_stats[cell_id] = (total_count, selected_count)

    if not R:
        # Throw Exception
        raise ValueError("Selected sample R is empty, cannot compute HPFR.")
        
    score, sum_psS, sum_psR, sum_rF= HPFR(R, optimal_psS, optimal_sS, W, K)

    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time, len(CL), cell_stats

def stratified_sampling(S: List[Place], k: int, W: float, G: int, optimal_psS, optimal_sS):

    t_prep_start = time.time()
    grid = SquareGrid(S, G)
    CL = grid.get_full_cells() # Get non-empty cells
    prep_time = time.time() - t_prep_start
        
    K = len(S)
    if K == 0 or k == 0:
        return [], 0.0, 0.0, 0.0, prep_time, 0.0, 0, {}
    
    R: List[Place] = []
    k_alloc: Dict[Tuple[int, int], int] = {} 
    cell_stats: Dict[Tuple[int, int], Tuple[int, int]] = {}
    remainders = [] 
    total_k_allocated = 0
    cell_map = {c.id: c for c in CL} 

    t_selection_start = time.time()
    
    for c in CL:
        if c.size() == 0:
            continue
        
        ideal = k * (c.size() / K)
        integer_part = math.floor(ideal)
        
        k_alloc[c.id] = integer_part
        total_k_allocated += integer_part
        remainders.append((c.id, ideal - integer_part))

    k_remaining = k - total_k_allocated
    
    remainders.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(min(k_remaining, len(remainders))):
        c_id_to_add = remainders[i][0]
        
        k_alloc[c_id_to_add] += 1
    
    for cell_id, num_to_pick in k_alloc.items():
        if num_to_pick > 0:
            cell = cell_map[cell_id]
            actual_pick = min(num_to_pick, cell.size())
            
            if actual_pick > 0:
                R.extend(random.sample(cell.places, actual_pick))

    selection_time = time.time() - t_selection_start
    
    
    for c in CL: # Use all non-empty cells
        cell_id = c.id
        total_count = c.size()
        selected_count = k_alloc.get(cell_id, 0) # Get from k_alloc
        cell_stats[cell_id] = (total_count, selected_count)

    if not R:
        # Throw Exception
        raise ValueError("Selected sample R is empty, cannot compute HPFR.")
        
    score, sum_psS, sum_psR, sum_rF = HPFR(R, optimal_psS, optimal_sS, W, K)

    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time, len(CL), cell_stats

def stratified_grid_sampling(S: List[Place], k: int, W: float, G: int, optimal_psS, optimal_sS):
    t_prep_start = time.time()
    grid = SquareGrid(S, G)
    CL = grid.get_full_cells()
    prep_time = time.time() - t_prep_start
    
    K = len(S)
    if K == 0 or k == 0:
        return [], 0.0, 0.0, 0.0, prep_time, 0.0, 0, {}

    R: List[Place] = []
    k_alloc: Dict[Tuple[int, int], int] = {}
    cell_stats: Dict[Tuple[int, int], Tuple[int, int]] = {}
    remainders = []
    cell_map = {c.id: c for c in CL} 

    cell_scores = {}
    total_grid_score = 0.0
    
    for c in CL:
        count = c.size()
        if count == 0:
            continue
            
        sum_weights = sum(p.rF for p in c.places)
        
        score = sum_weights
        
        cell_scores[c.id] = score
        total_grid_score += score
        
    t_selection_start = time.time()
    if total_grid_score == 0:
        return [], 0.0, 0.0, 0.0, prep_time, 0.0, 0, {}

    total_k_allocated = 0
    
    for c_id, score in cell_scores.items():
        ideal = k * (score / total_grid_score)
        
        integer_part = math.floor(ideal)
        k_alloc[c_id] = integer_part
        total_k_allocated += integer_part
        remainders.append((c_id, ideal - integer_part))

    k_remaining = k - total_k_allocated
    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(min(k_remaining, len(remainders))):
        c_id_to_add = remainders[i][0]
        k_alloc[c_id_to_add] += 1

    for cell_id, num_to_pick in k_alloc.items():
        if num_to_pick > 0:
            cell = cell_map[cell_id]
            actual_pick = min(num_to_pick, cell.size())
            
            if actual_pick > 0:
                R.extend(random.sample(cell.places, actual_pick))

    selection_time = time.time() - t_selection_start
    
    for c in CL:
        cell_id = c.id
        total_count = c.size()
        selected_count = k_alloc.get(cell_id, 0)
        cell_stats[cell_id] = (total_count, selected_count)

    if not R:
        raise ValueError("Selected sample R is empty.")
        
    score, sum_psS, sum_psR, sum_rF = HPFR(R, optimal_psS, optimal_sS, W, K)
    
    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time, len(CL), cell_stats

def grid_weighted_sampling(S: List[Place], k: int, W: float, G: int, optimal_psS, optimal_sS):
    """
    Weighted Grid Sampling: Allocates k proportional to total rF in cell.
    CRITICAL: Selects points with HIGHEST rF first (Greedy) to maximize score.
    """
    t_prep_start = time.time()
    grid = SquareGrid(S, G)
    CL = grid.get_full_cells()
    prep_time = time.time() - t_prep_start
    
    K = len(S)
    if K == 0 or k == 0:
        return [], 0.0, 0.0, 0.0, prep_time, 0.0, 0, {}

    R: List[Place] = []
    k_alloc: Dict[Tuple[int, int], int] = {}
    cell_stats: Dict[Tuple[int, int], Tuple[int, int]] = {}
    remainders = []
    cell_map = {c.id: c for c in CL} 

    cell_scores = {}
    total_grid_score = 0.0
    
    for c in CL:
        count = c.size()
        if count == 0:
            continue
        
        # Calculate weight of the cell based on rF of its points
        sum_weights = sum(p.rF for p in c.places)
        score = sum_weights
        
        cell_scores[c.id] = score
        total_grid_score += score
        
    t_selection_start = time.time()
    if total_grid_score == 0:
        return [], 0.0, 0.0, 0.0, prep_time, 0.0, 0, {}

    total_k_allocated = 0
    
    # 1. Allocation (Proportional to Weight)
    for c_id, score in cell_scores.items():
        ideal = k * (score / total_grid_score)
        integer_part = math.floor(ideal)
        k_alloc[c_id] = integer_part
        total_k_allocated += integer_part
        remainders.append((c_id, ideal - integer_part))

    k_remaining = k - total_k_allocated
    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(min(k_remaining, len(remainders))):
        c_id_to_add = remainders[i][0]
        k_alloc[c_id_to_add] += 1

    # 2. Selection (Greedy by rF)
    for cell_id, num_to_pick in k_alloc.items():
        if num_to_pick > 0:
            cell = cell_map[cell_id]
            actual_pick = min(num_to_pick, cell.size())
            
            if actual_pick > 0:
                # FIX: Sort by rF descending instead of random.sample
                # This ensures the 'Weighted' method actually picks the 'best' points
                sorted_places = sorted(cell.places, key=lambda p: p.rF, reverse=True)
                R.extend(sorted_places[:actual_pick])

    selection_time = time.time() - t_selection_start
    
    for c in CL:
        cell_id = c.id
        total_count = c.size()
        selected_count = k_alloc.get(cell_id, 0)
        cell_stats[cell_id] = (total_count, selected_count)

    if not R:
        raise ValueError("Selected sample R is empty.")
        
    score, sum_psS, sum_psR, sum_rF = HPFR(R, optimal_psS, optimal_sS, W, K)
    
    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time, len(CL), cell_stats

def quadtree_sampling(S: List[Place], k: int, W: float, m: int, d: int, optimal_psS, optimal_sS):

    t_prep_start = time.time()
    qtree = QuadTree(S, m, d)
    leaves = qtree.get_leaves()
    prep_time = time.time() - t_prep_start

    K = len(S)
    if K == 0 or k == 0:
        return [], 0.0, 0.0, 0.0, prep_time, 0.0, 0, {}

    t_selection_start = time.time()
    
    R: List[Place] = []
    k_alloc: Dict[int, int] = {}
    cell_stats: Dict[Tuple[int, int], Tuple[int, int]] = {}
    remainders = []
    total_k_allocated = 0
    
    leaf_map = {i: leaf for i, leaf in enumerate(leaves)}
    
    for i, leaf in leaf_map.items():
        leaf_size = len(leaf.places)
        if leaf_size == 0:
            continue
            
        ideal = k * (leaf_size / K)
        integer_part = math.floor(ideal)
        
        k_alloc[i] = integer_part
        total_k_allocated += integer_part
        remainders.append((i, ideal - integer_part))
        
    k_remaining = k - total_k_allocated
    remainders.sort(key=lambda x: x[1], reverse=True)
    
    for j in range(min(k_remaining, len(remainders))):
        idx_to_add = remainders[j][0]
        k_alloc[idx_to_add] += 1
        
    for i, num_to_pick in k_alloc.items():
        if num_to_pick > 0:
            leaf = leaf_map[i]
            actual_pick = min(num_to_pick, len(leaf.places))
            
            if actual_pick > 0:
                R.extend(random.sample(leaf.places, actual_pick))
                
    selection_time = time.time() - t_selection_start
    
    for i, leaf in leaf_map.items():
        total_count = len(leaf.places)
        selected_count = k_alloc.get(i, 0)
        cell_stats[(i, 0)] = (total_count, selected_count)

    if not R:
        raise ValueError("Selected sample R is empty, cannot compute HPFR.")
        
    score, sum_psS, sum_psR, sum_rF = HPFR(R, optimal_psS, optimal_sS, W, K)
    
    return R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time, len(leaves), cell_stats
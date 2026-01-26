import random
import time
from typing import Dict, Tuple
from alg.HPF_eq import HPFR, HPFR_no_r
from models import List, Place

def sampling(S: List[Place], k: int, W, psS, sS) -> Tuple[List[Place], Dict[int, float], float, float]:
    
    # Random selection
    R_sampling, pruning_time = select_random(S, k)
    
    # Compute final scores
    score, sum_psS, sum_psR, sum_rF = HPFR_no_r(R_sampling, psS, sS, W, len(S))
    
    return R_sampling, score, sum_psS, sum_psR, sum_rF, 0.0, pruning_time

def old_sampling(S: List[Place], k: int, W, psS, sS) -> Tuple[List[Place], Dict[int, float], float, float]:
    
    # Random selection
    R_sampling, pruning_time = select_random(S, k)
    
    # Compute final scores
    score, sum_psS, sum_psR, sum_rF = HPFR(R_sampling, psS, sS, W, len(S))
    
    return R_sampling, score, sum_psS, sum_psR, sum_rF, 0.0, pruning_time

def biased_sampling(S: List[Place], k: int, W, psS, sS) -> Tuple[List[Place], Dict[int, float], float, float]:
    
    # Random selection based on p.rF weights
    R_sampling, pruning_time = select_biased_random(S, k)
    
    # Compute final scores
    score, sum_psS, sum_psR, sum_rF = HPFR(R_sampling, psS, sS, W, len(S))
    
    return R_sampling, score, sum_psS, sum_psR, sum_rF, 0.0, pruning_time

def select_random(S: List[Place], k: int):
    
    pruning_time_start = time.time()
    sampled_S = random.sample(S, k)
    pruning_time = time.time() - pruning_time_start
    
    return  sampled_S, pruning_time

def select_biased_random(S: List[Place], k: int):
    
    pruning_time_start = time.time()
    sampled_S = random.choices(
        S,
        weights=[p.rF for p in S],
        k=k
    )
    pruning_time = time.time() - pruning_time_start
    
    return  sampled_S, pruning_time
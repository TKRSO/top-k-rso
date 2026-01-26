from baseline_iadu import base_precompute, baseline_iadu_algorithm
from biased_sampling import select_random
from grid_iadu import grid_based_iadu_algorithm, virtual_grid_based_algorithm
from models import List, Place, SquareGrid
from HPF_eq import HPFR, HPFR_div

################################################################################################################3
#####################################################################################################################
# --- Hybrid method ---
def hybrid(S: List[Place], k: int, K_sample, W, exact_psS, exact_sS):
    K = len(S)
    g = K/(k*W)
    if K_sample < k:
        raise ValueError(f"Hybrid error: K_sample ({K_sample}) is smaller than k ({k})")

    # pruning
    biased_sampled_S, pruning_time = select_random(S, K_sample)
    W_hybrid = K_sample / (k * g)
    
    # Preparation for hybrid
    bs_psS, bs_sS, prep_time = base_precompute(biased_sampled_S)
    # exact_psS, exact_sS, exact_prep_time = base_precompute(S)
    
    
    # Run baseline IAdU on sampled set
    R_hybrid, selection_time = baseline_iadu_algorithm(biased_sampled_S, K_sample, k, W_hybrid, bs_psS, bs_sS)
    
    # Compute final scores
    score, psS_sum, psR_sum, rF_sum = HPFR(R_hybrid, exact_psS, exact_sS, W, K)
    
    return R_hybrid, score, psS_sum, psR_sum, rF_sum, prep_time, selection_time, pruning_time, W_hybrid

def hybrid_on_grid(S: List[Place], k: int, G, K_sample, W, exact_psS, exact_sS):
    K = len(S)
    if K_sample < k:
        raise ValueError(f"Hybrid error: K_sample ({K_sample}) is smaller than k ({k})")
    g = K/(k*W)
    
    biased_sampled_S, pruning_time = select_random(S, K_sample)
    W_hybrid = K_sample / (k*g)
    
    # Preparation for hybrid
    grid = SquareGrid(biased_sampled_S, G)
    CL = grid.get_full_cells()
    bs_psS, bs_sS, prep_time = virtual_grid_based_algorithm(CL, biased_sampled_S)
    # exact_psS, exact_sS, exact_prep_time = base_precompute(S)
        
    # Run grid IAdU algorithm
    R_hybrid, selection_time = grid_based_iadu_algorithm(biased_sampled_S, CL, W_hybrid, bs_psS,  bs_sS, k)
    
    # Compute final scores
    score, sum_psS, sum_psR, sum_rF = HPFR(R_hybrid, exact_psS, exact_sS, W, K)

    return R_hybrid, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time, pruning_time
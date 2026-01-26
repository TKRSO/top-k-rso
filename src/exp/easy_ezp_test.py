import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Adjust path to find sibling directories (src/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from log.logger import ExperimentLogger
from log.runner import ExperimentRunner
from config import COMBO, GAMMAS, NUM_CELLS, DATASET_NAMES
from models import SquareGrid

# --- ALGORITHM IMPORTS ---
from alg.baseline_iadu import iadu, load_dataset, plot_selected
from alg.extension_sampling import grid_sampling, grid_weighted_sampling, quadtree_sampling

# --- PLOTTING LOGIC ---
pdf_pages = None 

def plot_experiment_results(S, shape, K, k, G, algo_results):
    global pdf_pages
    if pdf_pages is None: return

    try:
        grid = SquareGrid(S, G)
        lenCL = len(grid.get_full_cells())
    except:
        grid = None
        lenCL = "N/A"

    # CHANGED: 4 columns instead of 3 to include Quadtree
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.suptitle(f"Shape: {shape}  |  K={K}, k={k}  |  G={G}  |  lenCL={lenCL}", fontsize=16)
    
    # 1. Base IAdU
    if 'base_iadu' in algo_results:
        res = algo_results['base_iadu']
        plot_selected(S, res['R'], f"Base IAdU\nHPFR: {res['score']: .4f}", axes[0], grid=grid)
    else:
        axes[0].text(0.5, 0.5, "Base IAdU Failed", ha='center')

    # 2. Grid Standard
    if 'grid_standard' in algo_results:
        res = algo_results['grid_standard']
        full_tuple = res['raw_res']
        cell_stats = full_tuple[7] if len(full_tuple) > 7 else None
        plot_selected(S, res['R'], f"Grid Sampling\nHPFR: {res['score']: .4f}", axes[1], grid=grid, cell_stats=cell_stats)
    else:
        axes[1].text(0.5, 0.5, "Grid Standard Failed", ha='center')

    # 3. Grid Weighted
    if 'grid_weighted' in algo_results:
        res = algo_results['grid_weighted']
        full_tuple = res['raw_res']
        cell_stats = full_tuple[7] if len(full_tuple) > 7 else None
        plot_selected(S, res['R'], f"Grid Weighted\nHPFR: {res['score']: .4f}", axes[2], grid=grid, cell_stats=cell_stats)
    else:
        axes[2].text(0.5, 0.5, "Grid Weighted Failed", ha='center')

    # 4. Quadtree Sampling (NEW)
    if 'quadtree_sampling' in algo_results:
        res = algo_results['quadtree_sampling']
        full_tuple = res['raw_res']
        cell_stats = full_tuple[7] if len(full_tuple) > 7 else None
        # Note: Quadtree doesn't map perfectly to a SquareGrid visual, so we pass grid=None
        plot_selected(S, res['R'], f"Quadtree\nHPFR: {res['score']: .4f}", axes[3], grid=None, cell_stats=cell_stats)
    else:
        axes[3].text(0.5, 0.5, "Quadtree Failed", ha='center')

    pdf_pages.savefig(fig)
    plt.close(fig)

def run():
    global pdf_pages
    pdf_filename = "comparison_results.pdf"
    pdf_pages = PdfPages(pdf_filename)
    
    logger = ExperimentLogger("LogTest")
    runner = ExperimentRunner(load_dataset, logger, plot_callback=plot_experiment_results)

    # 3. Register Algorithms
    runner.register("base_iadu", iadu) 
    
    # Corrected: Removed params={'G': 'dynamic'}
    runner.register("grid_standard", grid_sampling)
    runner.register("grid_weighted", grid_weighted_sampling)
    
    runner.register(
        "quadtree_sampling", 
        quadtree_sampling, 
        params={'m': 80, 'd': 6}
    )

    print(f"Starting experiments on {len(DATASET_NAMES)} datasets...")
    try:
        runner.run_all(datasets=DATASET_NAMES, combos=COMBO, gammas=GAMMAS, G_values=NUM_CELLS)
    finally:
        pdf_pages.close()
        print(f"✔ PDF Plots saved to: {pdf_filename}")

if __name__ == "__main__":
    run()
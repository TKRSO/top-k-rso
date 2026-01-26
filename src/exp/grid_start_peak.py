import sys
import os
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Ensure parent directory is in path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from log.logger import ExperimentLogger
from log.runner import ExperimentRunner
from models import SquareGrid

# --- ALGORITHM IMPORTS ---
from alg.baseline_iadu import iadu, load_dataset, plot_selected
from alg.extension_sampling import grid_weighted_sampling

# --- CONFIGURATION ---
# We want to test if there is a peak, so we need a granular list of Grid Sizes (G).
# We will define G based on the square root of total cells (e.g., 2x2, 3x3 ... 20x20)
GRID_ROOTS = list(range(2, 21, 2)) + [25, 30, 40] # [2, 4, 6, ..., 20, 25, 30, 40]
NUM_CELLS = [r*r for r in GRID_ROOTS] # [4, 16, 36, ..., 1600]

# Standard Fixed Parameters
DATASET_NAMES = ["bubble", "flower", "s_curve"] # Representative shapes
COMBOS = [(1000, 20)] # Fixed K=1000, k=20 to isolate the variable G
GAMMAS = [2] # Fixed gamma for weight calculation

# Global PDF object
pdf_pages = None 

def plot_experiment_results(S, shape, K, k, G, algo_results):
    """
    Custom plotting callback to visualize Baseline vs Grid Weighted side-by-side
    """
    global pdf_pages
    if pdf_pages is None: return

    try:
        grid = SquareGrid(S, G)
        # Calculate stats for title
        filled_cells = len(grid.get_full_cells())
        total_cells = G
    except:
        grid = None
        filled_cells = "N/A"
        total_cells = G

    # Create a figure with 2 columns: Baseline vs Grid Weighted
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Dataset: {shape} | Grid Cells: {total_cells} ({int(math.sqrt(total_cells))}x{int(math.sqrt(total_cells))}) | Filled: {filled_cells}", fontsize=14)
    
    # 1. Base IAdU (Reference)
    if 'base_iadu' in algo_results:
        res = algo_results['base_iadu']
        plot_selected(S, res['R'], f"Base IAdU\nHPFR: {res['score']: .4f}", axes[0], grid=grid)
    else:
        axes[0].text(0.5, 0.5, "Base IAdU Failed", ha='center')

    # 2. Grid Weighted (Variable)
    if 'grid_weighted' in algo_results:
        res = algo_results['grid_weighted']
        full_tuple = res['raw_res']
        # Extract cell stats if available (index 7 in the return tuple)
        cell_stats = full_tuple[7] if len(full_tuple) > 7 else None
        
        # Calculate % difference from baseline for the title
        diff = 0
        if 'base_iadu' in algo_results:
            base_score = algo_results['base_iadu']['score']
            weighted_score = res['score']
            if base_score != 0:
                diff = (weighted_score - base_score) / base_score * 100
        
        title = f"Grid Weighted\nHPFR: {res['score']: .4f} ({diff:+.2f}%)"
        plot_selected(S, res['R'], title, axes[1], grid=grid, cell_stats=cell_stats)
    else:
        axes[1].text(0.5, 0.5, "Grid Weighted Failed", ha='center')

    pdf_pages.savefig(fig)
    plt.close(fig)

def run():
    global pdf_pages
    pdf_filename = "grid_size_peak_analysis.pdf"
    pdf_pages = PdfPages(pdf_filename)
    
    # Initialize Logger
    logger = ExperimentLogger("grid_size_analysis")
    
    # Initialize Runner
    runner = ExperimentRunner(load_dataset, logger, plot_callback=plot_experiment_results)

    # Register Algorithms
    runner.register("base_iadu", iadu) 
    runner.register("grid_weighted", grid_weighted_sampling)

    print(f"=== Starting Grid Size Peak Analysis ===")
    print(f"Datasets: {DATASET_NAMES}")
    print(f"Grid Sizes (Cells): {NUM_CELLS}")
    
    try:
        runner.run_all(datasets=DATASET_NAMES, combos=COMBOS, gammas=GAMMAS, G_values=NUM_CELLS)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pdf_pages.close()
        logger.save()
        print(f"\n✔ PDF Plots saved to: {pdf_filename}")
        print(f"✔ Excel Logs saved to: grid_size_analysis.xlsx")

if __name__ == "__main__":
    run()
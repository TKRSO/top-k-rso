import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
from log.logger import ExperimentLogger
from log.runner import ExperimentRunner
from log.plotter import ExperimentPlotter

from alg.baseline_iadu import iadu, load_dataset
from alg.grid_iadu import grid_iadu
from alg.biased_sampling import biased_sampling
from alg.extension_sampling import grid_sampling, stratified_grid_sampling

def run():
    plotter = ExperimentPlotter("score_vs_cardinality_plots.pdf")
    
    logger = ExperimentLogger("score_vs_cardinality_results", baseline_name="base_iadu")
    
    runner = ExperimentRunner(load_dataset, logger, plot_callback=plotter.plot_results)

    print("Registering algorithms...")
    
    runner.register("base_iadu", iadu)
    runner.register("grid_sampling", grid_sampling)
    runner.register("grid_stratified", stratified_grid_sampling)
    runner.register("biased_sampling", biased_sampling)

    print(f"=== Starting Experiment ===")
    print(f"Datasets: {cfg.DATASET_NAMES}")
    print(f"Combos: {cfg.COMBO}")
    print(f"Grid Sizes: {cfg.NUM_CELLS}")
    
    try:
        runner.run_all(
            datasets=cfg.DATASET_NAMES, 
            combos=cfg.COMBO, 
            gammas=cfg.GAMMAS, 
            G_values=cfg.NUM_CELLS
        )
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n!!! CRITICAL ERROR !!! {e}")
        import traceback
        traceback.print_exc()
    finally:
        plotter.close()
        logger.save()
        print(f"\n✔ PDF Plots saved to: results_comparison.pdf")
        print(f"✔ Excel Logs saved to: targeted_comparison.xlsx")

if __name__ == "__main__":
    run()
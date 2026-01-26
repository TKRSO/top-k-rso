import sys
import os
# Ensure parent directory is in path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
from log.logger import ExperimentLogger
from log.runner import ExperimentRunner
from log.plotter import ExperimentPlotter

# --- ALGORITHM IMPORTS ---
from alg.baseline_iadu import iadu_no_r, load_dataset
from alg.biased_sampling import sampling
from alg.extension_sampling import grid_sampling

def run():
    plotter = ExperimentPlotter("no_r_plots.pdf")
    
    logger = ExperimentLogger("no_rf_res", baseline_name="base_iadu_no_r")
    
    runner = ExperimentRunner(load_dataset, logger, plot_callback=plotter.plot_results)

    print("Registering algorithms...")
    
    runner.register("base_iadu_no_r", iadu_no_r)
    runner.register("grid_sampling", grid_sampling)
    runner.register("sampling", sampling)

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
        # Properly close plotter and save logs
        plotter.close()
        logger.save()

if __name__ == "__main__":
    run()
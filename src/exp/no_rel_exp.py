import sys
import os
# Ensure parent directory is in path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
from log.logger import ExperimentLogger
from log.runner import ExperimentRunner
from log.plotter import ExperimentPlotter
# --- ALGORITHM IMPORTS ---
from alg.baseline_iadu import iadu, iadu_no_r, load_dataset
from alg.grid_iadu import grid_iadu
from alg.biased_sampling import biased_sampling, sampling
from alg.extension_sampling import stratified_sampling, grid_sampling_no_r, grid_sampling


def run():
    # 1. Initialize Logger and Plotter
    # The plotter is now initialized once and will accumulate pages into the PDF
    logger = ExperimentLogger("var_K_norf", baseline_name="base_iadu_no_rF", aggregate_datasets=True)
    plotter = ExperimentPlotter("plots.pdf")

    # 2. Initialize Runner with the Plotter's callback
    # The updated ExperimentRunner passes (S, shape, K, k, G, W, wrf, algo_results)
    # The updated ExperimentPlotter.plot_results accepts exactly these arguments.
    runner = ExperimentRunner(load_dataset, logger, plot_callback=plotter.plot_results)

    print("Registering algorithms...")
    runner.register("base_iadu_no_rF", iadu_no_r)
        
    runner.register("stratified_sampling(no rF)", grid_sampling_no_r)
    
    runner.register("sampling", sampling)

    print(f"=== Starting Experiment ===")
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
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 3. Clean up and Save
        plotter.close()
        logger.save()

if __name__ == "__main__":
    run()
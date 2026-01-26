import sys
import os

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from log.logger import ExperimentLogger
from log.runner import ExperimentRunner
from log.plotter import ExperimentPlotter  

# --- ALGORITHMS ---
from alg.baseline_iadu import iadu, iadu_no_r, load_dataset
from alg.biased_sampling import biased_sampling
from alg.extension_sampling import grid_weighted_sampling, stratified_sampling
import config as cfg

def run():
    # 1. Initialize Logger and Plotter
    # The plotter is now initialized once and will accumulate pages into the PDF
    logger = ExperimentLogger("optical_test_results", baseline_name="base_iadu")
    plotter = ExperimentPlotter("optical_test_plots.pdf")
    runner = ExperimentRunner(load_dataset, logger, plot_callback=plotter.plot_results)

    print("Registering algorithms...")
    runner.register("base_iadu", iadu)
    runner.register("iadu_no_r", iadu_no_r)  # Example of registering another variant
    runner.register("stratified_sampling", stratified_sampling)
    runner.register("biased_sampling", biased_sampling)

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
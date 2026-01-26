import time
import inspect
from typing import Dict, List, Any, Callable

# Import the optimization target for precomputation
from alg.baseline_iadu import base_precompute
import config as cfg

class ExperimentRunner:
    def __init__(self, load_dataset_func, logger, plot_callback=None):
        self.load_dataset = load_dataset_func
        self.logger = logger
        self.algorithms = {}
        self.plot_callback = plot_callback 

    def register(self, name: str, func: Callable, params: Dict[str, Any] = None):
        self.algorithms[name] = {'func': func, 'params': params or {}}

    def run_all(self, datasets, combos, gammas, G_values):
        # 1. Outer Loop: Combos (K, k)
        for (K, k) in combos:
            
            # 2. Dataset Loop: Load S ONCE per combo
            for shape in datasets:
                S = self.load_dataset(shape, K)
                if not S:
                    print(f"  [Skipping] Dataset '{shape}' not found.")
                    continue

                # 3. Precompute Baseline ONCE
                print(f"  > Precomputing Exact Baseline stats for {shape}...")
                try:
                    exact_psS, exact_sS, exact_base_prep_time = base_precompute(S)
                    # Context used by all algorithms
                    base_context = {
                        'exact_psS': exact_psS, 
                        'exact_sS': exact_sS, 
                        'base_prep_time': exact_base_prep_time
                    }
                except Exception as e:
                    print(f"  ! Precompute failed: {e}")
                    continue

                # 4. Grid Size Loop
                for G in G_values:
                    # --- CACHING STEP ---
                    # Run W-Independent algorithms (e.g. Random, Spatial) ONCE here.
                    # We will reuse these results for every 'gamma'.
                    cached_results = {}
                    
                    # Identify which algos don't need W/g/wrf
                    w_independent_algos = []
                    w_dependent_algos = []

                    for name, algo_def in self.algorithms.items():
                        sig = inspect.signature(algo_def['func'])
                        # Check params for dependency on weights
                        if any(p in sig.parameters for p in ['W', 'g', 'wrf']):
                            w_dependent_algos.append(name)
                        else:
                            w_independent_algos.append(name)

                    # Run Independent Algos Now
                    if w_independent_algos:
                        print(f"  > Running static algos for G={G}...")
                        # We pass a dummy W=0, but filter ensures it isn't used
                        cached_results = self._run_batch(
                            w_independent_algos, S, k, 0, G, base_context
                        )

                    # 5. Gamma Loop (Weight variations)
                    for g in gammas:
                        W = g * K / k
                        print(f"\n=== Running Combo: {shape} | K={K}, k={k}, G={G}, W={W:.2f} ===")

                        # Initialize Row
                        row = {
                            "shape": shape, "K": K, "k": k, "W": W,
                            "wrf": cfg.wrf,
                            "g*K/k": f"{g}*K/k", "G": G, "lenCL": 0 
                        }

                        # Run Dependent Algos (Optimization methods that change with W)
                        current_results = self._run_batch(
                            w_dependent_algos, S, k, W, G, base_context
                        )

                        # Merge Cached Results into Current Results
                        current_results.update(cached_results)

                        # Log everything to the row
                        for name, res_data in current_results.items():
                            self._log_result_to_row(row, name, res_data)

                        self.logger.log(row)

                        if self.plot_callback:
                            self.plot_callback(S, shape, K, k, G, W, cfg.wrf, current_results)
        
        self.logger.save()

    def _run_batch(self, algo_names, S, k, W, G, context):
        """Helper to run a specific list of algorithms and return a dict of results."""
        results = {}
        for name in algo_names:
            algo_def = self.algorithms[name]
            res = self._execute_algo(name, algo_def, S, k, W, G, context)
            if res:
                results[name] = res
        return results

    def _execute_algo(self, name, algo_def, S, k, W, G, context):
        """Executes a single algorithm and returns the raw result dictionary."""
        func = algo_def['func']
        
        # Base Arguments
        available_args = {
            'S': S, 'k': k, 'W': W, 'G': G,
            "wrf": cfg.wrf,
            'exact_psS': context['exact_psS'],     
            'exact_sS': context['exact_sS'],       
            'prep_time': context['base_prep_time'],
            'psS': context['exact_psS'], 
            'sS': context['exact_sS'],
            'optimal_psS': context['exact_psS'], 
            'optimal_sS': context['exact_sS']
        }

        # Merge Algorithm Params
        available_args.update(algo_def.get('params', {}))

        # Filter Arguments via Signature
        sig = inspect.signature(func)
        call_kwargs = {k: v for k, v in available_args.items() if k in sig.parameters}
        
        try:
            res = func(**call_kwargs)
            
            # Unpacking
            # Tuple: (R, score, sum_psS, sum_psR, sum_rF, prep_time, selection_time, [lenCL])
            return {
                'R': res[0],
                'score': res[1],
                'stats': {
                    'pss_sum': res[2],
                    'psr_sum': res[3],
                    'rf_sum': res[4],
                    'prep_time': res[5] if name != "base_iadu" else context['base_prep_time'],
                    'sel_time': res[6],
                    'lenCL': res[7] if len(res) > 7 else 0
                }
            }
        except Exception as e:
            print(f"  Error running {name}: {e}")
            return None

    def _log_result_to_row(self, row, name, res_data):
        """Writes the result dictionary into the flat CSV row structure."""
        stats = res_data['stats']
        row[f"{name}_hpfr"] = res_data['score']
        row[f"{name}_pss_sum"] = stats['pss_sum']
        row[f"{name}_psr_sum"] = stats['psr_sum']
        row[f"{name}_rf_sum"] = stats['rf_sum']
        row[f"{name}_prep_time"] = stats['prep_time']
        row[f"{name}_sel_time"] = stats['sel_time']
        row[f"{name}_x_time"] = stats['prep_time'] + stats['sel_time']
        
        # Log lenCL if it exists and isn't 0 (updates global row if valid)
        if stats['lenCL'] > 0:
            row["lenCL"] = stats['lenCL']
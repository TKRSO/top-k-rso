import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from models import SquareGrid
from alg.baseline_iadu import plot_selected

class ExperimentPlotter:
    def __init__(self, filename="results.pdf"):
        self.filename = filename
        self.pdf = PdfPages(filename)
        print(f"   -> Plotter initialized. Saving to: {self.filename}")

    def plot_results(self, S, shape, K, k, G, W, wrf, algo_results):
        """
        Universal plotter that adapts to ANY number of algorithms.
        - Uses A4 Portrait page size.
        - Forces every subplot to be SQUARE.
        - Removes axis numbers (ticks) for a cleaner look.
        - Adds comprehensive settings info to the header.
        """
        if not algo_results:
            return

        # 1. Determine Grid Layout
        n = len(algo_results)
        if n == 1: cols = 1
        elif n == 2: cols = 2 
        elif n == 4: cols = 2 
        else: cols = min(n, 3) 
        
        rows = math.ceil(n / cols)

        # 2. Create Figure with A4 Dimensions (Portrait)
        # A4 is approx 8.27 x 11.69 inches
        A4_WIDTH = 8.27
        A4_HEIGHT = 11.69
        fig, axes = plt.subplots(rows, cols, figsize=(A4_WIDTH, A4_HEIGHT))
        
        # 3. Prepare Grid Object & Header Info
        grid_dims_str = "N/A"
        grid_obj = None
        try:
            if S is not None and G > 0:
                grid_obj = SquareGrid(S, G)
                Ax, Ay = grid_obj.dims()
                grid_dims_str = f"{Ax}x{Ay}"
        except Exception as e:
            print(f"Warning: Could not create visualization grid: {e}")

        # Comprehensive Header Info
        header_text = (
            f"Dataset: {shape}\n"
            f"K={K}  |  k={k}  |  W={W:.4f}  |  wrf={wrf}\n"
            f"Grid G={G} ({grid_dims_str})"
        )
        
        # Add Title with extra padding at the top
        fig.suptitle(header_text, fontsize=12, fontweight='bold', y=0.98, va='top')

        # 4. Flatten axes
        if n == 1: ax_list = [axes]
        else: ax_list = axes.flatten()

        # 5. Plot Each Algorithm
        for i, (algo_key, res) in enumerate(algo_results.items()):
            ax = ax_list[i]
            
            # --- TWEAK: REMOVE NUMBERS/TICKS ---
            ax.set_xticks([])
            ax.set_yticks([])
            
            # FORCE SQUARE ASPECT RATIO
            ax.set_aspect('equal', adjustable='box')
            
            display_title = algo_key.replace("_", " ").title()
            score_val = res.get('score', 0.0)
            
            # --- DRAW BACKGROUND GRID LINES (zorder=0) ---
            if grid_obj and S is not None:
                S_np = np.array(S)
                if S_np.ndim == 2 and S_np.shape[0] > 0:
                    min_x, min_y = S_np.min(axis=0)
                    max_x, max_y = S_np.max(axis=0)
                    Ax, Ay = grid_obj.dims()

                    # Vertical Lines
                    if Ax > 0:
                        xs = np.linspace(min_x, max_x, Ax + 1)
                        ax.vlines(x=xs, ymin=min_y, ymax=max_y, 
                                  colors='black', linestyles=':', linewidth=0.5, alpha=0.5, zorder=0)

                    # Horizontal Lines
                    if Ay > 0:
                        ys = np.linspace(min_y, max_y, Ay + 1)
                        ax.hlines(y=ys, xmin=min_x, xmax=max_x, 
                                  colors='black', linestyles=':', linewidth=0.5, alpha=0.5, zorder=0)

            # --- PLOT ALGORITHM ---
            try:
                cell_stats = None
                if 'raw_res' in res and isinstance(res['raw_res'], tuple):
                    if len(res['raw_res']) > 7:
                        cell_stats = res['raw_res'][7]

                plot_selected(
                    S, 
                    res['R'], 
                    f"{display_title}\nScore: {score_val:.4f}", 
                    ax, 
                    grid=grid_obj, 
                    cell_stats=cell_stats
                )
            except Exception as e:
                ax.text(0.5, 0.5, f"Error Plotting\n{e}", ha='center', color='red')
                ax.set_axis_off()

            # --- FORCE BORDERS ---
            ax.axis('on')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
                spine.set_zorder(100)

        # 6. Hide unused subplots
        for j in range(n, len(ax_list)):
            ax_list[j].axis('off')

        # 7. Adjust Layout
        plt.tight_layout(rect=[0, 0.02, 1, 0.90])
        
        self.pdf.savefig(fig)
        plt.close(fig)

    def close(self):
        if self.pdf:
            self.pdf.close()
            print(f"✔ PDF Plots saved/closed: {self.filename}")
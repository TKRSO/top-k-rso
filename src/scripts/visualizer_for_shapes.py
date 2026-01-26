import os
import sys
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List

# Ensure src directory is in path to import models and config
# This is crucial for unpickling the 'Place' objects
try:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from models import Place
    import config
except ImportError as e:
    print(f"Error: Could not import 'models' or 'config'. Make sure they are in the 'src' directory.")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

def visualize_datasets_to_pdf():
    """
    Loads all datasets specified in config.py from the ../datasets/ folder
    and generates a multi-page PDF visualizing each one as a scatter plot.
    """
    
    print("Starting dataset visualization...")
    
    # --- 1. Get Config ---
    try:
        k_values = sorted(list(set([k_val for k_val, _ in config.COMBO])))
        shape_names = config.DATASET_NAMES
    except Exception as e:
        print(f"Error reading 'COMBO' or 'DATASET_NAMES' from config.py: {e}")
        return

    if not k_values:
        print("No K values found in config.COMBO. Exiting.")
        return
    if not shape_names:
        print("No dataset names found in config.DATASET_NAMES. Exiting.")
        return

    # --- 2. Define Paths ---
    base_dir = os.path.dirname(__file__)
    datasets_dir = os.path.join(base_dir, "..", "datasets")
    output_pdf_path = os.path.join(base_dir, "..", "shape_datasets_visualization.pdf")

    if not os.path.isdir(datasets_dir):
        print(f"Error: Datasets directory not found at: {datasets_dir}")
        print("Please run 'generate_shapes.py' first.")
        return

    print(f"Loading datasets from: {datasets_dir}")
    print(f"Saving PDF to: {output_pdf_path}")

    # --- 3. Create PDF ---
    try:
        with PdfPages(output_pdf_path) as pdf:
            total_plots = 0
            
            # Loop over all shapes and K values from config
            for shape_name in shape_names:
                for K in k_values:
                    
                    data_filename = f"{shape_name}_K{K}.pkl"
                    data_path = os.path.join(datasets_dir, data_filename)
                    
                    if not os.path.exists(data_path):
                        print(f"  Skipping: File not found: {data_filename}")
                        continue
                    
                    print(f"  Plotting: {data_filename}")
                    
                    # Load the dataset
                    try:
                        with open(data_path, 'rb') as f:
                            dataset: List[Place] = pickle.load(f)
                    except Exception as e:
                        print(f"    Error loading pickle file {data_filename}: {e}")
                        continue
                        
                    # Extract coordinates
                    if not dataset:
                        print(f"    Warning: Dataset {data_filename} is empty.")
                        continue
                        
                    try:
                        x_coords = [place.coords[0] for place in dataset]
                        y_coords = [place.coords[1] for place in dataset]
                    except Exception as e:
                        print(f"    Error reading coordinates from Place object: {e}")
                        continue

                    # Create the plot
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(x_coords, y_coords, s=5, alpha=0.6) # s=5 for smaller dots
                    
                    title = f"{shape_name.replace('_', ' ').title()} (K={K})"
                    ax.set_title(title, fontsize=16)
                    ax.set_xlabel("X Coordinate")
                    ax.set_ylabel("Y Coordinate")
                    ax.set_aspect('equal', 'box') # Ensure aspect ratio is correct
                    ax.grid(True, linestyle='--', alpha=0.5)
                    
                    # Save the current figure to the PDF
                    pdf.savefig(fig)
                    
                    # Close the figure to free memory
                    plt.close(fig)
                    total_plots += 1
            
            print(f"\nSuccessfully created PDF with {total_plots} plots.")

    except Exception as e:
        print(f"An error occurred while creating the PDF: {e}")

if __name__ == "__main__":
    visualize_datasets_to_pdf()
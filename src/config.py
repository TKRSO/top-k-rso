# config.py
import math


#NUM_CELLS = [64, 100, 256, 529, 1024]  # Different grid sizes for experiments

#default G = 256
NUM_CELLS = [16]

RUN_X_TIMES = 1

COMBO = [
    # (100, 20),
    (200, 20),
    (500, 20),
    (500, 15),
    # (1000, 10),
    # (1000, 20),
    # (1000, 50),
    # (1000, 100),
    # (5000, 20),
    # (5000, 50),
    # (5000, 100)
]

GAMMAS = [1, 0.5, 0.65, 0.75]  # example values for g

wrf = 0.2  # scaling factor for rF in baseline_iadu

DATASET_NAMES = [
    "dbpedia_1994_FIFA_World_Cup_squads",
    # "dbpedia_1998_FIFA_World_Cup_squads",
    # "dbpedia_2002_FIFA_World_Cup_squads",
    # "dbpedia_2010_FIFA_World_Cup_squads",
]

'''

    (100, 20),
    (200, 20),
    (500, 20),
    (1000, 10),
    (1000, 15),
    (1000, 10),
    (1000, 20),
    (1000, 50),
    (1000, 100),
    (1000, 50),
    (5000, 20),
    (5000, 50),
    (5000, 100)
    
'''

SIMULATED_DATASETS = [
 "s_curve",
    "bubble",]

# Generate GRID_RANGE dynamically based on NUM_CELLS
def get_grid_range_for_cells(num_cells: int, cell_size: float = 1.0) -> tuple:
    G = int(math.sqrt(num_cells))
    return (0, G * cell_size)

CELL_SIZE = 1.0
GRID_RANGES = {g: get_grid_range_for_cells(g, CELL_SIZE) for g in NUM_CELLS}

# config.py
import math


#NUM_CELLS = [64, 100, 256, 529, 1024]  # Different grid sizes for experiments

#default G = 256
NUM_CELLS = [16]

RUN_X_TIMES = 2

COMBO = [
    # (100, 20),
    # (200, 20),
    # (500, 20),
    # (1000, 10),
    # (1000, 15),
    (1000, 20),
    # (1000, 50),
    # (1000, 100),
    # (5000, 20),
    # (5000, 50),
    # (5000, 100)
]

GAMMAS = [1,]  # example values for g  0.75, 0.65, 0.5, 0.25

wrf = 1  # scaling factor for rF in baseline_iadu

DATASET_NAMES = [
    "dbpedia_1994_FIFA_World_Cup_squads",
    # "dbpedia_1998_FIFA_World_Cup_squads",
    # "dbpedia_2002_FIFA_World_Cup_squads",
    # "dbpedia_2010_FIFA_World_Cup_squads",
    # "dbpedia_2012–13_Dayton_Flyers_men's_basketball_team",
    # "dbpedia_2012–13_UMass_Minutemen_basketball_team",
    # "dbpedia_2013–14_Oregon_Ducks_men's_basketball_team",
    # "dbpedia_2013–14_Tulsa_Golden_Hurricane_men's_basketball_team",
    # "dbpedia_List_of_Harvard_University_people",
    # "dbpedia_List_of_Phi_Beta_Sigma_chapters",
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

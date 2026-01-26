import os
import folium
import numpy as np
from typing import List
from baseline_iadu import load_dataset
from models import Place
import config as cfg


def get_center(S: List[Place]) -> tuple:
    coords = np.array([p.coords for p in S])
    lat, lon = coords[:, 0].mean(), coords[:, 1].mean()
    return float(lat), float(lon)


def plot_all_dbpedia_datasets():
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
    total = 0

    for K, k in cfg.COMBO:
        for G in cfg.NUM_CELLS:
            for region in cfg.DBPEDIA_DATASET_NAMES:
                fname = f"{region}_K{K}_k{k}_G{G}.pkl"
                path = os.path.join("datasets", fname)
                if not os.path.exists(path):
                    continue

                try:
                    S: List[Place] = load_dataset(region, K, k, G)
                    region_name = region.replace("dbpedia_", "").replace("_", " ")

                    # Plot all points
                    for p in S:
                        lat, lon = float(p.coords[0]), float(p.coords[1])
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=2,
                            color="#3399FF",
                            fill=True,
                            fill_color="#3399FF",
                            fill_opacity=0.5,
                            weight=0.6
                        ).add_to(m)

                    # Add label at region center
                    lat_c, lon_c = get_center(S)
                    folium.map.Marker(
                        [lat_c, lon_c],
                        icon=folium.DivIcon(html=f"""<div style="font-size:10pt; color:black"><b>{region_name}</b></div>""")
                    ).add_to(m)

                    total += 1
                except Exception as e:
                    print(f"✗ Failed: {region}: {e}")

    print(f"✔ Done: plotted {total} DBpedia datasets")
    m.save("maps/dbpedia_all_regions.html")
    print("✔ Saved map to maps/dbpedia_all_regions.html")


if __name__ == "__main__":
    os.makedirs("maps", exist_ok=True)
    plot_all_dbpedia_datasets()

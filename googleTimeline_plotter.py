import json
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from collections import defaultdict

# Paths to spatial data
COASTLINE_PATH = "data/ne_10m_coastline/ne_10m_coastline.shp"
ROADS_PATH = "data/ne_10m_roads_north_america/ne_10m_roads_north_america.shp"

def load_shapefile(path):
    """Load a shapefile and handle errors."""
    try:
        return gpd.read_file(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def parse_timeline(json_file):
    """Extract fine-grained GPS paths from Google Takeout timeline data."""
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    daily_routes = defaultdict(list)

    for entry in data:
        if "timelinePath" in entry:
            date = entry["startTime"].split("T")[0]  # Extract YYYY-MM-DD
            route_points = [
                tuple(map(float, point["point"].replace("geo:", "").split(",")))
                for point in entry["timelinePath"]
            ]
            daily_routes[date].append(route_points)  # Store full trajectory

    return daily_routes

def get_global_bounds(daily_routes, margin=0):
    """Compute max bounding box across all routes to ensure consistent aspect ratio."""
    all_lats, all_lons = zip(
        *[(lat, lon) for routes in daily_routes.values() for route in routes for lat, lon in route]
    )

    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)

    lat_range, lon_range = lat_max - lat_min, lon_max - lon_min
    return (
        lat_min - lat_range * margin,
        lat_max + lat_range * margin,
        lon_min - lon_range * margin,
        lon_max + lon_range * margin,
    )

def get_dynamic_bounds(routes, margin=0):
    """Compute bounding box for each day's routes."""
    lats, lons = zip(*[coord for route in routes for coord in route])
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    lat_range, lon_range = lat_max - lat_min, lon_max - lon_min
    return (
        lat_min - lat_range * margin,
        lat_max + lat_range * margin,
        lon_min - lon_range * margin,
        lon_max - lon_range * margin,
    )

def create_frames(daily_routes, output_folder="frames", dynamic_extent=False, add_coastline=False, add_roads=False, aspect_ratio="1:1"):
    """Generates high-resolution frames using full timelinePath data."""
    os.makedirs(output_folder, exist_ok=True)

    coastline = load_shapefile(COASTLINE_PATH) if add_coastline else None
    roads = load_shapefile(ROADS_PATH) if add_roads else None
    global_bounds = get_global_bounds(daily_routes)

    # Aspect ratio dictionary
    aspect_ratios = {
        "1:1": (6, 6),
        "9:16": (9, 16),
        "16:9": (16, 9),
        "4:5": (4, 5),
        "3:4": (3, 4),
        "2:3": (2, 3),
    }

    if aspect_ratio not in aspect_ratios:
        print(f"Warning: Aspect ratio '{aspect_ratio}' not recognized. Defaulting to 1:1.")
        aspect_ratio = "1:1"

    fig_size = aspect_ratios[aspect_ratio]

    for date, routes in sorted(daily_routes.items()):
        fig, ax = plt.subplots(figsize=fig_size, dpi=150)

        if dynamic_extent:
            bounds = get_dynamic_bounds(routes)
        else:
            bounds = global_bounds

        lat_min, lat_max, lon_min, lon_max = bounds

        # Compute ranges
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        # **Prevent ZeroDivisionError**
        if lon_range == 0:  
            lon_range = max(lat_range * 0.01, 0.0001)  # Small buffer
            lon_min -= lon_range / 2
            lon_max += lon_range / 2

        if lat_range == 0:
            lat_range = max(lon_range * 0.01, 0.0001)  # Small buffer
            lat_min -= lat_range / 2
            lat_max += lat_range / 2

        desired_aspect = fig_size[1] / fig_size[0]  # Height / Width

        if lat_range / lon_range > desired_aspect:
            # Expand longitude range to match aspect ratio
            center_lon = (lon_min + lon_max) / 2
            lon_range = lat_range / desired_aspect
            lon_min, lon_max = center_lon - lon_range / 2, center_lon + lon_range / 2
        else:
            # Expand latitude range to match aspect ratio
            center_lat = (lat_min + lat_max) / 2
            lat_range = lon_range * desired_aspect
            lat_min, lat_max = center_lat - lat_range / 2, center_lat + lat_range / 2

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Remove axes, labels, and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # add bg
        fig.patch.set_facecolor("#080808")  # entire bg
        ax.set_facecolor("#080808")  # matchy

        # Plot coastline first to avoid re-rendering on each route plot
        if add_coastline and coastline is not None:
            coastline.plot(color="#aaaaaa", alpha=0.666, linewidth=0.5, ax=ax)

        # Plot roads
        if add_roads and roads is not None:
            clipped_roads = roads.cx[lon_min:lon_max, lat_min:lat_max]
            if not clipped_roads.empty:
                clipped_roads.plot(color="#797979", alpha=0.5, linewidth=0.35, ax=ax)

        # **Efficiently plot the full trajectory**
        for route_points in routes:
            route_array = np.array(route_points)
            ax.plot(route_array[:, 1], route_array[:, 0], "w-", linewidth=1.2, alpha=0.9)  # High-precision paths

        # Add date title
        ax.set_title(date, color="#ffffff", alpha=0.8, family="monospace", fontsize=24, fontweight="normal", stretch="ultra-expanded", loc="left", y=0, pad=0)

        # Save frame with consistent resolution
        frame_path = os.path.join(output_folder, f"{date}.png")
        fig.savefig(frame_path, dpi=150, bbox_inches="tight", pad_inches=0, transparent=False, format="png")
        plt.close(fig)

    print(f"Frames saved in {output_folder}")

def main(json_file, output_folder="frames", dynamic_extent=False, add_coastline=False, add_roads=False, aspect_ratio="1:1"):
    """Generates daily route frames from Google Takeout Timeline JSON."""
    daily_routes = parse_timeline(json_file)
    create_frames(daily_routes, output_folder=output_folder, dynamic_extent=dynamic_extent, add_coastline=add_coastline, add_roads=add_roads, aspect_ratio=aspect_ratio)
    print(f"Frames saved in '{output_folder}'")

# Run with roads & 10m coastline enabled
if __name__ == "__main__":
    json_path = "data/location-history_20250130.json"
    main(json_path, output_folder="output/googlePlots/two", dynamic_extent=True, add_coastline=True, add_roads=False, aspect_ratio="9:16")
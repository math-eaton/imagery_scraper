import json
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from collections import defaultdict
from shapely.geometry import box

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

def get_bounding_box(routes, margin=0.02):
    """Compute a unified bounding box from the timeline paths with a margin."""
    if not routes:
        return None  # Return None for empty movement days

    all_lats, all_lons = zip(*[coord for route in routes for coord in route])
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    # Apply margin symmetrically
    lat_min -= lat_range * margin
    lat_max += lat_range * margin
    lon_min -= lon_range * margin
    lon_max += lon_range * margin

    return lat_min, lat_max, lon_min, lon_max

def create_frames(daily_routes, output_folder="frames", add_coastline=False, add_roads=False, aspect_ratio="1:1", margin=0.02, dpi=300):
    """Generates high-resolution frames ensuring aspect ratio conformity before clipping spatial layers."""
    os.makedirs(output_folder, exist_ok=True)

    # Load spatial data
    coastline = load_shapefile(COASTLINE_PATH) if add_coastline else None
    roads = load_shapefile(ROADS_PATH) if add_roads else None

    # Aspect ratio dictionary (output dimensions in inches for DPI scaling)
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

    fig_size = aspect_ratios[aspect_ratio]  # Size in inches
    desired_aspect = fig_size[1] / fig_size[0]  # Height / Width ratio

    # **Store last valid bounding box (for no-movement frames)**
    last_valid_bounds = None

    for date, routes in sorted(daily_routes.items()):
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        # **Step 1: Compute bounding box from timeline paths**
        bbox = get_bounding_box(routes, margin)

        if bbox is None:  # No movement, reuse last valid bounding box
            if last_valid_bounds is None:
                print(f"Skipping {date}: No previous valid bounding box available.")
                plt.close(fig)
                continue
            else:
                bbox = last_valid_bounds
        else:
            last_valid_bounds = bbox  # Update last valid bounding box

        lat_min, lat_max, lon_min, lon_max = bbox

        # **Step 2: Adjust bounding box to conform with aspect ratio**
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        # **Prevent ZeroDivisionError**
        min_buffer = 0.0001
        if lon_range == 0:
            lon_min -= min_buffer
            lon_max += min_buffer
            lon_range = lon_max - lon_min
        if lat_range == 0:
            lat_min -= min_buffer
            lat_max += min_buffer
            lat_range = lat_max - lat_min

        # **Expand bounding box to match the aspect ratio BEFORE clipping**
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        if lat_range / lon_range > desired_aspect:
            # Expand longitude range to match aspect ratio
            lon_range = lat_range / desired_aspect
            lon_min, lon_max = center_lon - lon_range / 2, center_lon + lon_range / 2
        else:
            # Expand latitude range to match aspect ratio
            lat_range = lon_range * desired_aspect
            lat_min, lat_max = center_lat - lat_range / 2, center_lat + lat_range / 2

        # **Step 3: Clip all spatial layers using the adjusted bounding box**
        bounding_box = box(lon_min, lat_min, lon_max, lat_max)
        print(date)
        print(bounding_box)
        clipped_roads = gpd.clip(roads, bounding_box) if add_roads and roads is not None else None
        clipped_coastline = gpd.clip(coastline, bounding_box) if add_coastline and coastline is not None else None

        # **Step 4: Set plot extent using the adjusted bounding box**
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Remove axes, labels, and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # Add dark background
        fig.patch.set_facecolor("#080808")
        ax.set_facecolor("#080808")

        # **Step 5: Plot clipped coastline**
        if clipped_coastline is not None and not clipped_coastline.empty:
            clipped_coastline.plot(color="#aaaaaa", alpha=0.5, linewidth=0.666, ax=ax)

        # **Step 6: Plot clipped roads**
        if clipped_roads is not None and not clipped_roads.empty:
            clipped_roads.plot(color="#696969", alpha=0.85, linewidth=0.333, ax=ax)

        # **Step 7: Plot the full trajectory**
        for route_points in routes:
            route_array = np.array(route_points)
            ax.plot(route_array[:, 1], route_array[:, 0], "w-", linewidth=1.2, alpha=0.9)

        # **Step 8: Add date title**
        ax.set_title(
            date,
            color="#ffffff",
            alpha=0.8,
            family="monospace",
            fontsize=24,
            fontweight="normal",
            stretch="ultra-expanded",
            loc="left",
            y=0,
            pad=-0,
        )

        # **Step 9: Save the frame while enforcing aspect ratio**
        frame_path = os.path.join(output_folder, f"{date}.png")
        # nb: plt.gca() to conform aspect
        plt.gca().set_aspect('auto')
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight", pad_inches=.666, transparent=False, format="png")
        # fig.savefig(frame_path, dpi=dpi, pad_inches=0, transparent=True, format="png")
        plt.close(fig)

    print(f"Frames saved in {output_folder}")

def main(json_file, output_folder="frames", dynamic_extent=False, add_coastline=False, add_roads=False, aspect_ratio="1:1", margin=0.02, dpi=150):
    """Generates daily route frames from Google Takeout Timeline JSON."""
    daily_routes = parse_timeline(json_file)
    create_frames(
        daily_routes,
        output_folder=output_folder,
        add_coastline=add_coastline,
        add_roads=add_roads,
        aspect_ratio=aspect_ratio,
        margin=margin,  # Pass the margin
        dpi=dpi,        # Pass the dpi
    )
    print(f"Frames saved in '{output_folder}'")

# Run with roads & 10m coastline enabled
if __name__ == "__main__":
    json_path = "data/location-history_20250130.json"
    main(json_path, output_folder="output/googlePlots/eight", add_coastline=True, add_roads=True, aspect_ratio="9:16", margin=0.1, dpi=150)
    
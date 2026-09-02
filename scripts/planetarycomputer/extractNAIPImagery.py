import argparse
import io
import math
import os
import re
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

PC_KEY = os.environ.get("planetary_computer_key")

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DATA_API_BASE = "https://planetarycomputer.microsoft.com/api/data/v1"
COLLECTION = "naip"

METERS_PER_DEGREE_LAT = 111320.0
EQUATOR_METERS_PER_PIXEL_AT_ZOOM_0 = 156543.03392

SESSION = requests.Session()
if PC_KEY:
    SESSION.headers["Ocp-Apim-Subscription-Key"] = PC_KEY

# Recipes lifted directly from Microsoft's documented NAIP render configuration:
# NAIP's 4-band "image" asset is band-ordered Red(1), Green(2), Blue(3), NIR(4).
BAND_MODES = {
    "color-infrared": {
        "assets": "image",
        "asset_bidx": "image|4,1,2",
        "color_formula": "Sigmoidal RGB 15 0.35",
    },
    "true-color": {
        "assets": "image",
        "asset_bidx": "image|1,2,3",
    },
    "nir-only": {
        "assets": "image",
        "asset_bidx": "image|4",
    },
    "ndvi": {
        "expression": "(image_b4-image_b1)/(image_b4+image_b1)",
        "rescale": "-1,1",
        "colormap_name": "rdylgn",
    },
}


def create_directories(*dirs):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def bbox_for_zoom(latitude, longitude, zoom_level, width_px, height_px):
    """Build a lon/lat bounding box around a point sized to match the ground footprint
    a standard slippy-map viewer would show at the given zoom level.

    Same Web Mercator ground-resolution formula the sibling scripts use in the other
    direction (meters_per_pixel = 156543.03392 * cos(lat) / 2^zoom), inverted here to
    go from a chosen zoom + output size to a crop extent in degrees.
    """
    lat_rad = math.radians(latitude)
    meters_per_pixel = EQUATOR_METERS_PER_PIXEL_AT_ZOOM_0 * math.cos(lat_rad) / (2**zoom_level)

    half_width_m = (width_px * meters_per_pixel) / 2
    half_height_m = (height_px * meters_per_pixel) / 2

    meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(lat_rad)
    dlat = half_height_m / METERS_PER_DEGREE_LAT
    dlon = half_width_m / meters_per_degree_lon

    return longitude - dlon, latitude - dlat, longitude + dlon, latitude + dlat


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def find_naip_item(latitude, longitude, datetime_filter=None):
    body = {
        "collections": [COLLECTION],
        "intersects": {"type": "Point", "coordinates": [longitude, latitude]},
        "limit": 1,
        "sortby": [{"field": "datetime", "direction": "desc"}],
    }
    if datetime_filter:
        body["datetime"] = datetime_filter

    response = SESSION.post(STAC_SEARCH_URL, json=body)
    response.raise_for_status()
    features = response.json().get("features", [])
    return features[0] if features else None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_naip_crop_image(latitude, longitude, unique_id, zoom_level, width, height, band_mode, image_format, datetime_filter, output_dir):
    unique_id = re.sub(r"[^\w-]+", "_", str(unique_id)).strip("_")

    item = find_naip_item(latitude, longitude, datetime_filter)
    if item is None:
        print(f"No NAIP coverage for {unique_id} at ({latitude}, {longitude}); skipping.")
        return

    min_lon, min_lat, max_lon, max_lat = bbox_for_zoom(latitude, longitude, zoom_level, width, height)
    geojson_feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
        },
    }

    params = {"collection": COLLECTION, "item": item["id"]}
    params.update(BAND_MODES[band_mode])

    crop_url = f"{DATA_API_BASE}/item/crop/{width}x{height}.{image_format}"
    response = SESSION.post(crop_url, params=params, json=geojson_feature)

    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        output_file_path = os.path.join(output_dir, f"{unique_id}.{image_format}")
        image.save(output_file_path)
    else:
        print(f"Failed to get NAIP crop for {unique_id}: {response.status_code} {response.content}")


def process_point_mode(df, args, output_dir):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        unique_id = f"{row[args.id_col]}_{row[args.name_col]}"

        get_naip_crop_image(
            row[args.lat_col],
            row[args.lon_col],
            unique_id,
            row[args.zoom_col],
            args.width,
            args.height,
            args.band_mode,
            args.image_format,
            args.datetime_filter,
            output_dir,
        )


def main(args):
    current_date = datetime.now().strftime("%Y%m%d")
    output_dir = f"{args.output_dir}_{current_date}"
    create_directories(output_dir)

    df = pd.read_csv(args.input_file)
    df = df.dropna(subset=[args.lat_col, args.lon_col, args.zoom_col])

    process_point_mode(df, args, output_dir)

    print("Process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NAIP aerial/NIR imagery from Microsoft Planetary Computer for points from a CSV file")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--lat-col", default="lat", help="Latitude column name")
    parser.add_argument("--lon-col", default="long", help="Longitude column name")
    parser.add_argument("--zoom-col", default="zoom", help="Per-row zoom level column name, sets the crop footprint size")
    parser.add_argument("--id-col", default="id", help="Unique identifier column name")
    parser.add_argument("--name-col", default="name", help="Human-readable name column name")
    parser.add_argument("--output-dir", default="output/unprocessed/naip_point", help="Output directory for images")
    parser.add_argument("--band-mode", default="color-infrared", choices=list(BAND_MODES), help="How to render NAIP's 4-band (R,G,B,NIR) data: color-infrared (NIR-R-G false color, highlights vegetation), true-color (R-G-B), nir-only (grayscale NIR band), or ndvi")
    parser.add_argument("--image-format", default="png", choices=["png", "jpeg", "tif", "webp"], help="Returned image format")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels")
    parser.add_argument("--datetime-filter", default=None, help="Optional STAC datetime filter (e.g. '2020-01-01/2020-12-31') to pick a specific NAIP vintage instead of the most recent coverage")
    args = parser.parse_args()
    main(args)

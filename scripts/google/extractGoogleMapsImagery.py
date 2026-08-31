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

MAPS_KEY = os.environ["maps_key"]

STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"

SQFT_TO_SQM = 0.09290304
EQUATOR_METERS_PER_PIXEL_AT_ZOOM_0 = 156543.03392


def create_directories(*dirs):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def zoom_for_facility_size(sqft_value, latitude, base_size_px, frame_padding, min_zoom, max_zoom, default_zoom):
    """Pick a zoom level so the frame covers roughly frame_padding x the facility's footprint.

    Derived from the standard Web Mercator ground-resolution formula:
    meters_per_pixel = 156543.03392 * cos(lat) / 2^zoom. We solve for the zoom whose
    resulting frame width (base_size_px * meters_per_pixel) is >= the padded facility
    footprint, so the whole facility stays in frame. Floors (rather than rounds) the
    zoom for the same reason: rounding up could crop the facility.
    """
    if sqft_value is None:
        return default_zoom

    try:
        sqft = float(str(sqft_value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default_zoom

    if not sqft or sqft <= 0:
        return default_zoom

    side_m = math.sqrt(sqft * SQFT_TO_SQM)
    frame_width_m = side_m * frame_padding
    meters_per_pixel_needed = frame_width_m / base_size_px

    lat_rad = math.radians(latitude)
    zoom = math.floor(math.log2(EQUATOR_METERS_PER_PIXEL_AT_ZOOM_0 * math.cos(lat_rad) / meters_per_pixel_needed))

    return max(min_zoom, min(max_zoom, zoom))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_google_static_image(latitude, longitude, unique_id, zoom_level, map_size, scale, output_dir):
    unique_id = re.sub(r"[^\w-]+", "_", str(unique_id)).strip("_")

    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom_level,
        "size": map_size,
        "scale": scale,
        "maptype": "satellite",
        "format": "png",
        "key": MAPS_KEY,
    }

    response = requests.get(STATIC_MAPS_URL, params=params)

    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        output_file_path = os.path.join(output_dir, f"{unique_id}.png")
        image.save(output_file_path)
    else:
        print(f"Failed to get map image for {unique_id}: {response.status_code} {response.content}")


def process_point_mode(df, args, output_dir):
    base_size_px = int(args.map_size.lower().split("x")[0])
    has_size_col = args.size_col in df.columns

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        unique_id = f"{index}_{row[args.id_col]}"

        sqft_value = row[args.size_col] if has_size_col and pd.notnull(row[args.size_col]) else None
        zoom_level = zoom_for_facility_size(
            sqft_value,
            row[args.lat_col],
            base_size_px,
            args.frame_padding,
            args.min_zoom,
            args.max_zoom,
            args.zoom_level,
        )

        get_google_static_image(
            row[args.lat_col],
            row[args.lon_col],
            unique_id,
            zoom_level,
            args.map_size,
            args.scale,
            output_dir,
        )


def main(args):
    current_date = datetime.now().strftime("%Y%m%d")
    output_dir = f"{args.output_dir}_{current_date}"
    create_directories(output_dir)

    df = pd.read_csv(args.input_file)
    df = df.dropna(subset=[args.lat_col, args.lon_col])

    process_point_mode(df, args, output_dir)

    print("Process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Google Maps satellite images for points from a CSV file")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--lat-col", default="lat", help="Latitude column name")
    parser.add_argument("--lon-col", default="long", help="Longitude column name")
    parser.add_argument("--id-col", default="facility_name", help="Unique identifier column name")
    parser.add_argument("--output-dir", default="output/unprocessed/point", help="Output directory for images")
    parser.add_argument("--map-size", default="640x640", help="Static map size in pixels before scale (max 640x640)")
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2], help="Scale factor (2 doubles actual returned pixel dimensions)")
    parser.add_argument("--zoom-level", type=int, default=15, help="Fallback zoom level when facility size is blank/unparseable")
    parser.add_argument("--size-col", default="facility_size_sqft", help="Facility size column (sqft) used to pick zoom level")
    parser.add_argument("--frame-padding", type=float, default=3.0, help="Frame width as a multiple of the facility's footprint side length")
    parser.add_argument("--min-zoom", type=int, default=10, help="Lowest zoom level to use for very large facilities")
    parser.add_argument("--max-zoom", type=int, default=20, help="Highest zoom level to use for very small facilities")
    args = parser.parse_args()
    main(args)

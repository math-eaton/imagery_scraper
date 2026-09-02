import argparse
import html
import math
import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

MAPS_KEY = os.environ["azure_maps_key"]

ATTRIBUTION_URL = "https://atlas.microsoft.com/map/attribution"
API_VERSION = "2024-04-01"

# microsoft.imagery only supports zoom 0-19 for the static image endpoint
# (must match the ceiling extractAzureMapsImagery.py used when the images were made).
IMAGERY_MAX_ZOOM = 19

SQFT_TO_SQM = 0.09290304
TILE_SIZE = 256

SESSION = requests.Session()

FILENAME_RE = re.compile(r"^(\d+)_")
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}

JOIN_COLUMNS = ["facility_name", "city", "state", "status", "facility_size_sqft"]


def zoom_for_facility_size(sqft_value, latitude, base_size_px, frame_padding, min_zoom, max_zoom, default_zoom):
    """Mirrors extractAzureMapsImagery.py's zoom pick so attribution bounds match the rendered frame."""
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
    equator_meters_per_pixel_at_zoom_0 = 156543.03392
    zoom = math.floor(math.log2(equator_meters_per_pixel_at_zoom_0 * math.cos(lat_rad) / meters_per_pixel_needed))

    return max(min_zoom, min(max_zoom, zoom))


def latlon_to_pixel_xy(latitude, longitude, zoom):
    """Bing/Azure Maps tile-system projection (https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system)."""
    lat = max(min(latitude, 85.05112878), -85.05112878)
    sin_lat = math.sin(math.radians(lat))
    x = (longitude + 180.0) / 360.0
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
    map_size = TILE_SIZE * (2**zoom)
    return x * map_size, y * map_size, map_size


def pixel_xy_to_latlon(pixel_x, pixel_y, map_size):
    x = (pixel_x / map_size) - 0.5
    y = 0.5 - (pixel_y / map_size)
    latitude = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
    longitude = 360 * x
    return latitude, longitude


def compute_image_bounds(latitude, longitude, zoom, width_px, height_px):
    """Bounding box of a static image centered on (lat, lon), in the
    "sw_lon,sw_lat,ne_lon,ne_lat" order the attribution API's bounds param expects."""
    cx, cy, map_size = latlon_to_pixel_xy(latitude, longitude, zoom)
    lat_max, lon_min = pixel_xy_to_latlon(cx - width_px / 2, cy - height_px / 2, map_size)
    lat_min, lon_max = pixel_xy_to_latlon(cx + width_px / 2, cy + height_px / 2, map_size)
    return lon_min, lat_min, lon_max, lat_max


def clean_citation_text(copyright_strings):
    cleaned = []
    for entry in copyright_strings:
        text = re.sub(r"<[^>]+>", "", entry)
        text = html.unescape(text).strip()
        if text:
            cleaned.append(text)
    return cleaned


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_map_attribution(tileset_id, zoom, bounds):
    params = {
        "api-version": API_VERSION,
        "subscription-key": MAPS_KEY,
        "tilesetId": tileset_id,
        "zoom": zoom,
        "bounds": ",".join(f"{v:.6f}" for v in bounds),
    }

    response = SESSION.get(ATTRIBUTION_URL, params=params)

    if response.status_code == 200:
        return response.json().get("copyrights", [])

    print(f"Failed to get attribution for zoom={zoom} bounds={bounds}: {response.status_code} {response.content}")
    return []


def parse_index_from_filename(filename):
    stem, ext = os.path.splitext(filename)
    ext = ext.lstrip(".").lower()
    if ext not in IMAGE_EXTENSIONS:
        return None

    match = FILENAME_RE.match(stem)
    if not match:
        return None

    return int(match.group(1)), stem


def main(args):
    if args.tileset_id == "microsoft.imagery" and args.max_zoom > IMAGERY_MAX_ZOOM:
        print(f"Note: clamping --max-zoom to {IMAGERY_MAX_ZOOM} (microsoft.imagery's supported ceiling).")
        args.max_zoom = IMAGERY_MAX_ZOOM

    df = pd.read_csv(args.input_file)
    df = df.dropna(subset=[args.lat_col, args.lon_col])
    has_size_col = args.size_col in df.columns

    image_files = sorted(os.listdir(args.image_dir))

    rows = []
    for filename in tqdm(image_files):
        parsed = parse_index_from_filename(filename)
        if parsed is None:
            continue
        index, unique_id = parsed

        if index not in df.index:
            print(f"Skipping {filename}: row index {index} not found in {args.input_file}")
            continue

        row = df.loc[index]
        latitude, longitude = row[args.lat_col], row[args.lon_col]

        sqft_value = row[args.size_col] if has_size_col and pd.notnull(row[args.size_col]) else None
        zoom_level = zoom_for_facility_size(
            sqft_value,
            latitude,
            args.map_width,
            args.frame_padding,
            args.min_zoom,
            args.max_zoom,
            args.zoom_level,
        )

        bounds = compute_image_bounds(latitude, longitude, zoom_level, args.map_width, args.map_height)
        copyrights = get_map_attribution(args.tileset_id, zoom_level, bounds)

        rows.append(
            {
                "image_uid": unique_id,
                "image_filename": filename,
                "lat": latitude,
                "long": longitude,
                "zoom_level": zoom_level,
                "tileset_id": args.tileset_id,
                "citation_text": "; ".join(clean_citation_text(copyrights)),
                "citation_raw": " | ".join(copyrights),
                **{col: row[col] if col in df.columns else None for col in JOIN_COLUMNS},
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a row-per-image CSV of Azure Maps imagery citations, joined to source facility attributes"
    )
    parser.add_argument("image_dir", help="Directory of images produced by extractAzureMapsImagery.py")
    parser.add_argument("input_file", help="Path to the same input CSV used to scrape the imagery")
    parser.add_argument("--output-csv", default=None, help="Output CSV path (default: <image_dir>_attribution.csv)")
    parser.add_argument("--lat-col", default="lat", help="Latitude column name")
    parser.add_argument("--lon-col", default="long", help="Longitude column name")
    parser.add_argument("--size-col", default="facility_size_sqft", help="Facility size column (sqft) used to pick zoom level")
    parser.add_argument("--tileset-id", default="microsoft.imagery", choices=["microsoft.imagery", "microsoft.base.road", "microsoft.base.darkgrey"], help="Azure Maps tileset the images were rendered from")
    parser.add_argument("--map-width", type=int, default=1280, help="Image width in pixels used at scrape time")
    parser.add_argument("--map-height", type=int, default=1280, help="Image height in pixels used at scrape time")
    parser.add_argument("--zoom-level", type=int, default=15, help="Fallback zoom level when facility size is blank/unparseable")
    parser.add_argument("--frame-padding", type=float, default=3.0, help="Frame width as a multiple of the facility's footprint side length")
    parser.add_argument("--min-zoom", type=int, default=10, help="Lowest zoom level used for very large facilities")
    parser.add_argument("--max-zoom", type=int, default=IMAGERY_MAX_ZOOM, help=f"Highest zoom level used for very small facilities (clamped to {IMAGERY_MAX_ZOOM} for microsoft.imagery)")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = f"{args.image_dir.rstrip('/')}_attribution.csv"

    main(args)

import argparse
import pandas as pd
import requests
import os
import io
import config
from PIL import Image
from process_imagery_halftone import process_image
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

def create_directories(*dirs):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def process_and_save_image(image, output_file_path, city, zoom_level):
    processed_image = process_image(image, output_file_path)
    if processed_image is not None:
        processed_image.save(output_file_path)
    else:
        print(f"Image for {city} at zoom level {zoom_level} was not processed due to low resolution.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_bing_map_image(center_latitude, center_longitude, city, zoom_level, map_style, map_size):
    base_url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/"
    params = {
        "center": f"{center_latitude},{center_longitude}",
        "zoomlevel": zoom_level,
        "mapSize": map_size,
        "format": "png",
        "key": config.bing_api_key,
    }
    full_url = base_url + map_style + "/" + params["center"] + "/" + str(params["zoomlevel"]) + "?mapSize=" + params["mapSize"] + "&format=" + params["format"] + "&key=" + params["key"]
    response = requests.get(full_url)

    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        crop_percentage = 20
        crop_pixel = int((crop_percentage/100) * image.height)
        left = 0
        top = 0
        right = image.width
        bottom = image.height - crop_pixel
        square_size = min(right, bottom)
        left = (image.width - square_size) / 2
        right = left + square_size
        image = image.crop((left, top, right, bottom))
        output_file_path = os.path.join(args.output_dir, f"{city}_{zoom_level}.png")
        process_and_save_image(image, output_file_path, city, zoom_level)
    else:
        print(f"Failed to get map image for {city} at zoom level {zoom_level}: {response.content}")

def main(args):
    create_directories(args.output_dir)
    df = pd.read_csv(args.input_file)
    unique_coords = df[['city', 'lon', 'lat']].drop_duplicates()
    print(unique_coords)

    for index, row in tqdm(unique_coords.iterrows(), total=unique_coords.shape[0]):
        lat = row['lat']
        lon = row['lon']
        city = row['city']
        for zoom_level in range(3, 20):
            get_bing_map_image(lat, lon, city, zoom_level, args.map_style, args.map_size)
    print("done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and extract images from Bing Maps API")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output-dir", default="output/tour_2024", help="Output directory for images")
    parser.add_argument("--map-style", default="Aerial", help="Bing Maps style (e.g., Aerial)")
    parser.add_argument("--map-size", default="1280,1280", help="Map size in pixels (e.g., 500,500)")
    args = parser.parse_args()
    main(args)

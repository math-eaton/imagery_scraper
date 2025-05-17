import argparse
import pandas as pd
import requests
import os
import io
import config
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
from tqdm import tqdm

# uncomment based on preferred processing algorithm
# from process_imagery_floydSteinberg import process_image
from process_imagery_halftone import process_image
# from process_imagery_ordered import process_image

current_date = datetime.now().strftime("%Y%m%d")

# Function to create directories if they don't exist
def create_directories(*dirs):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def process_and_save_image(image, unprocessed_output_file_path, processed_output_file_path, unique_id):
    # Process the image
    processed_image = process_image(image, processed_output_file_path)

    # If the image was processed successfully, save it
    if processed_image is not None:
        # Save the processed image
        processed_image.save(processed_output_file_path)
    else:
        print(f"Image for unique_id {unique_id} was not processed due to low resolution.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_bing_map_image_area(min_latitude, min_longitude, max_latitude, max_longitude, unique_id, map_style, map_size, angle, unprocessed_output_dir_area, processed_output_dir_area):
    # Ensure angle is a float and constrain to [0, 360]
    angle = int(round(float(angle) % 360))  # Safeguard against invalid values and round to nearest integer

    # Ensure unique_id is a string and remove any trailing .0 if present
    unique_id = str(int(float(unique_id)))

    # Define the base URL for the Bing Maps Static API
    base_url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/"

    # Define the parameters for the API request, including rotation using the 'dir' parameter
    params = {
        "mapArea": f"{min_latitude},{min_longitude},{max_latitude},{max_longitude}",
        "mapSize": map_size,
        "format": "png",
        "dir": angle,  # Specify the camera rotation angle here
        "key": config.bing_api_key,
    }


    # Create the full URL for the API request
    full_url = base_url + map_style + "?" + "&".join(f"{key}={value}" for key, value in params.items())

    # Make the API request
    response = requests.get(full_url)

    # Check that the request was successful
    if response.status_code == 200:
        # Open the image using PIL
        image = Image.open(io.BytesIO(response.content))

        # Define the dimensions for the crop
        crop_percentage = 20  # percentage to crop from the bottom
        crop_pixel = int((crop_percentage / 100) * image.height)  # calculate the number of pixels to crop

        left = 0
        top = 0
        right = image.width
        bottom = image.height - crop_pixel  # subtract the crop pixels from the height

        # Adjust left and right to maintain square aspect ratio
        square_size = min(right, bottom)  # size of the square is the smaller of width and height
        left = (image.width - square_size) / 2
        right = left + square_size

        # Crop the image
        image = image.crop((left, top, right, bottom))

        # Define the output file paths
        unprocessed_output_file_path = os.path.join(unprocessed_output_dir_area, f"{unique_id}.png")
        processed_output_file_path = os.path.join(processed_output_dir_area, f"{unique_id}.png")

        # Process and save the image
        process_and_save_image(image, unprocessed_output_file_path, processed_output_file_path, unique_id)
    else:
        print(f"Failed to get map image: {response.content}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_bing_map_image_point(center_latitude, center_longitude, unique_id, map_style, map_size, zoom_level, angle, unprocessed_output_dir_point, processed_output_dir_point):
    # Ensure angle is a valid integer between 0 and 360
    angle = int(round(float(angle) % 360))  # Safeguard against invalid values and round to nearest integer

    # Ensure unique_id is a string and remove any trailing .0 if present
    unique_id = str(int(float(unique_id)))


    # Define the base URL for the Bing Maps Static API
    base_url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/"

    # Define the parameters for the API request, including rotation using the 'dir' parameter
    params = {
        "center": f"{center_latitude},{center_longitude}",
        "zoomlevel": zoom_level,
        "mapSize": map_size,
        "format": "png",
        "dir": angle,  # Specify the camera rotation angle here
        "key": config.bing_api_key,
    }

    # Create the full URL for the API request
    full_url = base_url + map_style + "/" + params["center"] + "/" + str(params["zoomlevel"]) + "?mapSize=" + params["mapSize"] + "&format=" + params["format"] + "&dir=" + str(params["dir"]) + "&key=" + params["key"]

    # Make the API request
    response = requests.get(full_url)

    # Check that the request was successful
    if response.status_code == 200:
        # Open the image using PIL
        image = Image.open(io.BytesIO(response.content))

        # Define the dimensions for the crop
        crop_percentage = 20  # percentage to crop from the bottom
        crop_pixel = int((crop_percentage / 100) * image.height)  # calculate the number of pixels to crop

        left = 0
        top = 0
        right = image.width
        bottom = image.height - crop_pixel  # subtract the crop pixels from the height

        # Adjust left and right to maintain square aspect ratio
        square_size = min(right, bottom)  # size of the square is the smaller of width and height
        left = (image.width - square_size) / 2
        right = left + square_size

        # Crop the image
        image = image.crop((left, top, right, bottom))

        # Define the output file paths
        unprocessed_output_file_path = os.path.join(unprocessed_output_dir_point, f"{unique_id}.png")
        processed_output_file_path = os.path.join(processed_output_dir_point, f"{unique_id}.png")

        # Process and save the image
        process_and_save_image(image, unprocessed_output_file_path, processed_output_file_path, unique_id)
    else:
        print(f"Failed to get map image: {response.content}")


def process_point_mode(df, args, unprocessed_output_dir_point, processed_output_dir_point):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Download the image for the specific point (central coordinates)
        center_latitude = row['center_latitude']
        center_longitude = row['center_longitude']
        unique_id = row['unique_id']
        
        # Get the angle, default to 0 if not found
        angle = row['angle'] if pd.notnull(row['angle']) else 0

        # Call Bing Maps API to get the image for a point
        get_bing_map_image_point(
            center_latitude, 
            center_longitude, 
            unique_id, 
            args.map_style, 
            args.map_size, 
            args.zoom_level, 
            angle,  # Pass the angle to rotate the camera view
            unprocessed_output_dir_point, 
            processed_output_dir_point
        )


def process_area_mode(df, args, unprocessed_output_dir_area, processed_output_dir_area):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        min_latitude = 90
        max_latitude = -90
        min_longitude = 180
        max_longitude = -180

        # Loop over each column to extract coordinates
        for i in range(360):
            if ',' in row[str(i)]:
                try:
                    latitude, longitude = map(float, row[str(i)].split(','))
                    min_latitude = min(min_latitude, latitude)
                    max_latitude = max(max_latitude, latitude)
                    min_longitude = min(min_longitude, longitude)
                    max_longitude = max(max_longitude, longitude)
                except ValueError:
                    pass

        # Get the angle, default to 0 if not found
        angle = row['angle'] if pd.notnull(row['angle']) else 0

        # Download the image for the bounding box (area)
        get_bing_map_image_area(
            min_latitude, 
            min_longitude, 
            max_latitude, 
            max_longitude, 
            row['unique_id'], 
            args.map_style, 
            args.map_size, 
            angle,  # Pass the angle for the area-based map
            unprocessed_output_dir_area, 
            processed_output_dir_area
        )

def main(args):
    current_date = datetime.now().strftime("%Y%m%d")

    # Modify directory names to append the current date
    if args.mode == "area":
        unprocessed_output_dir_area = args.unprocessed_output_dir_area + "_" + current_date
        processed_output_dir_area = args.processed_output_dir_area + "_" + current_date
        create_directories(unprocessed_output_dir_area, processed_output_dir_area)
    elif args.mode == "point":
        unprocessed_output_dir_point = args.unprocessed_output_dir_point + "_" + current_date
        processed_output_dir_point = args.processed_output_dir_point + "_" + current_date
        create_directories(unprocessed_output_dir_point, processed_output_dir_point)

    # Load the data
    df = pd.read_csv(args.input_file)

    # Call the appropriate processing function based on the selected mode
    if args.mode == "area":
        process_area_mode(df, args, unprocessed_output_dir_area, processed_output_dir_area)
    elif args.mode == "point":
        process_point_mode(df, args, unprocessed_output_dir_point, processed_output_dir_point)

    print("Process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Bing Maps images for specific points or areas from a CSV file")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--mode", choices=["area", "point"], required=True, help="Choose the mode: area (based on extents) or point (based on central coordinates)")
    parser.add_argument("--unprocessed-output-dir-area", default="output/unprocessed/area", help="Output directory for unprocessed area images")
    parser.add_argument("--processed-output-dir-area", default="output/processed/area", help="Output directory for processed area images")
    parser.add_argument("--unprocessed-output-dir-point", default="output/unprocessed/point", help="Output directory for unprocessed point images")
    parser.add_argument("--processed-output-dir-point", default="output/processed/point", help="Output directory for processed point images")
    parser.add_argument("--map-style", default="Aerial", help="Bing Maps style (e.g., Aerial)")
    parser.add_argument("--map-size", default="500,500", help="Map size in pixels (e.g., 500,500)")
    parser.add_argument("--zoom-level", type=int, default=15, help="Bing Maps zoom level (only for point mode)")
    args = parser.parse_args()
    main(args)

import pandas as pd
import requests
import os
import io
import config
from PIL import Image
from process_imagery import process_image

# map config
map_size = "500,500"
map_style = "Aerial"
zoom_level = 15

unprocessed_output_dir_area = "output/bing_imagery/area"
processed_output_dir_area = "output/processed_imagery/area"
unprocessed_output_dir_point = "output/bing_imagery/point"
processed_output_dir_point = "output/processed_imagery/point"

# Create directories if they don't exist
os.makedirs(unprocessed_output_dir_area, exist_ok=True)
os.makedirs(processed_output_dir_area, exist_ok=True)
os.makedirs(unprocessed_output_dir_point, exist_ok=True)
os.makedirs(processed_output_dir_point, exist_ok=True)


def process_and_save_image(image, unprocessed_output_file_path, processed_output_file_path, application_id):
    # Save the unprocessed image
    # image.save(unprocessed_output_file_path)

    # Process the image
    processed_image = process_image(unprocessed_output_file_path, processed_output_file_path)

    # If the image was processed successfully, save it
    if processed_image is not None:
        # Save the processed image
        processed_image.save(processed_output_file_path)
    else:
        print(f"Image for application_id {application_id} was not processed due to low resolution.")


def get_bing_map_image_area(min_latitude, min_longitude, max_latitude, max_longitude, application_id):
    # Define the base URL for the Bing Maps Static API
    base_url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/"

    # Define the parameters for the API request
    params = {
        "mapArea": f"{min_latitude},{min_longitude},{max_latitude},{max_longitude}",
        "mapSize": map_size,
        "format": "png",
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

        # Define the output file paths
        unprocessed_output_file_path = os.path.join(unprocessed_output_dir_area, f"{application_id}.png")
        processed_output_file_path = os.path.join(processed_output_dir_area, f"{application_id}.png")

        # Process and save the image
        process_and_save_image(image, unprocessed_output_file_path, processed_output_file_path, application_id)
    else:
        print(f"Failed to get map image: {response.content}")


def get_bing_map_image_point(center_latitude, center_longitude, application_id):
    # Define the base URL for the Bing Maps Static API
    base_url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/"

    # Define the parameters for the API request
    params = {
        "center": f"{center_latitude},{center_longitude}",
        "zoomlevel": zoom_level,
        "mapSize": map_size,
        "format": "png",
        "key": config.bing_api_key,
    }

    # Create the full URL for the API request
    full_url = base_url + map_style + "/" + params["center"] + "/" + str(params["zoomlevel"]) + "?mapSize=" + params["mapSize"] + "&format=" + params["format"] + "&key=" + params["key"]

    # Make the API request
    response = requests.get(full_url)

    # Check that the request was successful
    if response.status_code == 200:
        # Open the image using PIL
        image = Image.open(io.BytesIO(response.content))

        # Define the output file paths
        unprocessed_output_file_path = os.path.join(unprocessed_output_dir_point, f"{application_id}.png")
        processed_output_file_path = os.path.join(processed_output_dir_point, f"{application_id}.png")

        # Process and save the image
        process_and_save_image(image, unprocessed_output_file_path, processed_output_file_path, application_id)
    else:
        print(f"Failed to get map image: {response.content}")


# Read the CSV file and get the first X rows for debugging purposes
df = pd.read_csv('creative_coding/data/fm_contours_sample.csv').head(50)

# read the full csv
# df = pd.read_csv('creative_coding/data/fm_contours_sample.csv')

# Loop over each row in the DataFrame
for index, row in df.iterrows():
    # Initialize min and max coordinates for each row
    min_latitude = 90
    max_latitude = -90
    min_longitude = 180
    max_longitude = -180

    # Loop over each column in the row
    for i in range(360):
        # Check if the column value is a valid latitude/longitude pair
        if ',' in row[str(i)]:
            try:
                # Get the coordinates from the column
                latitude, longitude = map(float, row[str(i)].split(','))

                # Update min and max coordinates
                min_latitude = min(min_latitude, latitude)
                max_latitude = max(max_latitude, latitude)
                min_longitude = min(min_longitude, longitude)
                max_longitude = max(max_longitude, longitude)
            except ValueError:
                # Ignore the column if it's not a valid latitude/longitude pair
                pass

    # Download the image for these coordinates
    get_bing_map_image_area(min_latitude, min_longitude, max_latitude, max_longitude, row['application_id'])

    # Get the coordinates from the 'transmitter_site' column
    # Assuming the column contains strings like '47.6097,-122.3331'
    center_latitude, center_longitude = map(float, row['transmitter_site'].split(','))

    # Download the image for these coordinates
    get_bing_map_image_point(center_latitude, center_longitude, row['application_id'])

print("done.")

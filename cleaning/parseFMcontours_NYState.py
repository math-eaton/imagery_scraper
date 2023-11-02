import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# Load New York State boundary from the GeoJSON file
ny_boundary_gdf = gpd.read_file('./data/raw/NYS_boundary.geojson')
# Combine all the polygons in MultiPolygon to one Polygon (if there are multiple polygons)
ny_boundary = ny_boundary_gdf.unary_union

# Function to check if a point is in New York
def is_in_new_york(coords):
    try:
        # Ensure coords is a string and not NaN or float
        if isinstance(coords, str):
            lat, lon = map(float, coords.strip('"').split(','))
            point = Point(lon, lat)
            return ny_boundary.contains(point)
    except (ValueError, AttributeError) as e:
        # If there's an error in conversion, it's not a valid coordinate
        return False
    return False


# Define the input and output CSV file names
input_csv = './data/raw/FM_service_contour_current.csv'
output_csv = './data/raw/FM_service_contour_NYS.csv'

# Get the total number of rows for the progress bar
total_rows = sum(1 for row in open(input_csv, 'r'))

# Define the chunk size
chunk_size = 10000  # Adjust based on your system's memory

# Initialize CSV writer and write the header
with open(output_csv, 'w', newline='') as outfile:
    writer = None

    # Initialize the progress bar
    pbar = tqdm(total=total_rows, desc='Processing', unit='row', ascii=True)

    # Process the input CSV in chunks
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, dtype=str, iterator=True):
        # Filter rows where the coordinates are within New York
        new_york_rows = chunk[chunk['transmitter_site'].apply(is_in_new_york)]
        
        # Remove the last '^' character from each row if it exists
        new_york_rows.replace({'\^': ''}, regex=True, inplace=True)
        
        # If the writer is not initialized, write the header and initialize it
        if writer is None:
            new_york_rows.to_csv(outfile, index=False)
            writer = True
        else:
            new_york_rows.to_csv(outfile, mode='a', header=False, index=False)

        # Update the progress bar
        pbar.update(chunk_size)

    pbar.close()

print("Processing complete. New York data saved to", output_csv)

import pandas as pd
import matplotlib.pyplot as plt
import os
from shapely.geometry import LineString
from tqdm import tqdm

# Function to simplify a list of coordinates
def simplify_polyline(coordinates, tolerance=0.01):
    line = LineString(coordinates)
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    return list(simplified_line.coords)

# Create the output directory if it doesn't exist
output_dir = 'output/polygon_images'
os.makedirs(output_dir, exist_ok=True)

# Specify the chunk size for reading the CSV
chunksize = 100

# Get the total number of rows in the CSV file
total_rows = sum(1 for _ in open('data/FM_service_contour_current.csv'))  # Count rows without reading the entire file
total_chunks = total_rows // chunksize + 1  # Calculate the total number of chunks

# Load the data in chunks and track progress with tqdm
for chunk_idx, df_chunk in tqdm(enumerate(pd.read_csv('data/FM_service_contour_current.csv', chunksize=chunksize, nrows=200)), total=total_chunks, desc='Total CSV'):
    chunk_rows = len(df_chunk)  # Number of rows in the current chunk
    for _, row in tqdm(df_chunk.iterrows(), total=chunk_rows, desc='Processing Chunk', leave=False):
        # Get columns named '0', '1', ..., '359' and drop missing values
        coordinates = row[[str(i) for i in range(360)]].dropna()
        # Convert strings "latitude, longitude" into a list of [latitude, longitude]
        coordinates = [coord.split(',') for coord in coordinates]
        # Transpose the list of coordinates into two lists: latitudes and longitudes
        latitudes, longitudes = [], []
        for coord in coordinates:
            try:
                latitude, longitude = float(coord[0]), float(coord[1])
                latitudes.append(latitude)
                longitudes.append(longitude)
            except (ValueError, IndexError):
                continue

        # Create a list of coordinates
        locations = list(zip(latitudes, longitudes))

        # Simplify the polyline
        simplified_locations = simplify_polyline(locations, tolerance=0.001)

        # Check if simplified_locations is empty before creating the plot
        if simplified_locations:
            # Create a new figure
            plt.figure(figsize=(5, 5), dpi=100)

            # Plot the filled polygon
            xs, ys = zip(*simplified_locations)
            plt.fill(xs, ys, color='blue', alpha=0.5)  # You can adjust the color and alpha (transparency) as needed

            # Remove axes
            plt.axis('off')

            # Set the extent of the plot to match the extent of the polyline feature
            min_x, min_y = min(xs), min(ys)
            max_x, max_y = max(xs), max(ys)
            plt.xlim(min_x, max_x)
            plt.ylim(min_y, max_y)

            # Save the figure as a PNG image with the name based on the application_id
            application_id = row['application_id']
            plt.savefig(f'output/polygon_images/{application_id}.png', transparent=True)

            # Close the figure to free up memory
            plt.close()

import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from tqdm import tqdm
import numpy as np

# Function to simplify a list of coordinates
def simplify_polyline(coordinates, tolerance=0.01):
    line = LineString(coordinates)
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    return list(simplified_line.coords)

# Function to get the number of lines in a file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Specify the input file path
input_file_path = 'data/processed/FM_service_contour_current_processed.csv'

# Get the total number of rows in the CSV file
len_of_file = file_len(input_file_path)
print(len_of_file)

# Skipping every Nth row
N = 10  # Change this value to your desired sampling rate
skipped = np.setdiff1d(np.arange(len_of_file), np.arange(0, len_of_file, N))
print(skipped)

# Specify the chunk size for reading the CSV
chunksize = 100

# Get the total number of chunks
total_chunks = len(skipped) // chunksize + 1

# Read the CSV file with skipped rows and chunking
for chunk_idx, df_chunk in tqdm(enumerate(pd.read_csv(input_file_path, skiprows=skipped, chunksize=chunksize)), total=total_chunks, desc='Total CSV'):
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

            # Plot the polyline
            xs, ys = zip(*simplified_locations)
            plt.plot(xs, ys, color='blue', linewidth=2)

            buffer = 0.001  # You can adjust this value

            # Set the extent of the plot to match the extent of the polyline feature
            min_x, min_y = min(xs), min(ys)
            max_x, max_y = max(xs), max(ys)

            # Add the buffer
            plt.xlim(min_x - buffer, max_x + buffer)
            plt.ylim(min_y - buffer, max_y + buffer)

            # Remove axes
            plt.axis('off')

            # Adjust the figure size to fit the plot
            plt.tight_layout(pad=0)

            # Save the figure as a PNG image with the name based on the application_id
            application_id = row['application_id']
            plt.savefig(f'output/polyline_images/{application_id}.png')

import argparse
import os
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

def main(args):
    # Get the total number of rows in the CSV file
    len_of_file = file_len(args.input_file)
    print(len_of_file)

    # Sample every Nth row
    N = args.sample_rate  # sample rate
    skipped = np.setdiff1d(np.arange(len_of_file), np.arange(0, len_of_file, N))
    print(skipped)

    df = pd.read_csv(args.input_file, skiprows=skipped)

    # Sort the DataFrame by 'application_id' in ascending order
    df = df.sort_values(by='application_id')

    # Specify the chunk size for reading the CSV
    chunksize = args.chunk_size

    # Get the total number of chunks
    total_chunks = len(skipped) // chunksize + 1

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read the CSV file with skipped rows and chunking
    for chunk_idx, df_chunk in tqdm(enumerate(pd.read_csv(args.input_file, skiprows=skipped, chunksize=chunksize)), total=total_chunks, desc='Total CSV'):
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
                # Add the first point again to close the circle
                simplified_locations.append(simplified_locations[0])

                # Create a new figure
                plt.figure(figsize=(8, 8), dpi=100)

                # Plot the polyline
                xs, ys = zip(*simplified_locations)
                plt.plot(xs, ys, color='black', linewidth=0.6)

                # Calculate the center point of the original polyline
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)

                # Define the scaling factors
                scaling_factors = np.linspace(1.0, 0.01, num=20)  # You can adjust the number of contours

                for scaling_factor in scaling_factors:
                    # Calculate the scaled coordinates
                    scaled_xs = [(x - center_x) * scaling_factor + center_x for x in xs]
                    scaled_ys = [(y - center_y) * scaling_factor + center_y for y in ys]

                    # Plot a scaled polyline to give an offset effect
                    plt.plot(scaled_xs, scaled_ys, color='black', linewidth=1)

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

                # Construct the output file path including the application_id
                application_id = row['application_id']
                output_path = os.path.join(args.output_dir, f'{application_id}.png')

                # Save the figure as a PNG image
                plt.savefig(output_path, transparent=True)

                # Close the figure to free up memory
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and plot polyline data from a CSV file")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--sample-rate", type=int, default=10, help="Sample rate for skipping rows")
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size for reading the CSV")
    parser.add_argument("--output-dir", default="scraping/satellite_imagery/output/polyline", help="Output directory for saving PNG files")
    args = parser.parse_args()
    main(args)
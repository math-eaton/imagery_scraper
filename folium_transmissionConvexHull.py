import pandas as pd
import folium
from shapely.geometry import LineString
from tqdm import tqdm

# Function to simplify a list of coordinates
def simplify_polyline(coordinates, tolerance=0.01):
    line = LineString(coordinates)
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    return list(simplified_line.coords)

# Create a Map instance
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5, tiles='CartoDB Positron')

# Specify the chunk size for reading the CSV
chunksize = 1000

# Get the total number of rows in the CSV file
total_rows = sum(1 for _ in open('data/FM_service_contour_current.csv'))  # Count rows without reading the entire file
total_chunks = total_rows // chunksize + 1  # Calculate the total number of chunks

# Load the data in chunks and track progress with tqdm
# for chunk_idx, df_chunk in tqdm(enumerate(pd.read_csv('data/FM_service_contour_current.csv', chunksize=chunksize, nrows=2000)), total=total_chunks, desc='Total CSV'):
for chunk_idx, df_chunk in tqdm(enumerate(pd.read_csv('data/FM_service_contour_current.csv', chunksize=chunksize, nrows=2000)), total=total_chunks, desc='Total CSV'):
    chunk_rows = len(df_chunk)  # Number of rows in the current chunk
    for _, row in tqdm(df_chunk.iterrows(), total=chunk_rows, desc='Processing Chunk', leave=False):
        # get columns named '0', '1', ..., '359' and drop missing values
        coordinates = row[[str(i) for i in range(360)]].dropna()
        # convert strings "latitude, longitude" into list of [latitude, longitude]
        coordinates = [coord.split(',') for coord in coordinates]
        # transpose list of coordinates into two lists: latitudes and longitudes
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
        
        # Check if simplified_locations is empty before creating a PolyLine
        if simplified_locations:
            # Create a PolyLine with the simplified coordinates:
            folium.PolyLine(locations=simplified_locations, color='#0c13e5', weight=1, opacity=0.7).add_to(m)

# Save it as html
m.save('output/folium/output.html')

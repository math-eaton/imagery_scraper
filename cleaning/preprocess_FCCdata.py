import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Read the CSV file
file_path = 'data/FM_service_contour_current.csv'
data = pd.read_csv(file_path)

# Convert the "application_id" column to a consistent data type (e.g., string)
data['application_id'] = data['application_id'].astype(str)

# Sort the data by the "application_id" column
data.sort_values(by='application_id', inplace=True)

# Extract latitude and longitude from the "transmitter_site" column
data['latitude'], data['longitude'] = data['transmitter_site'].str.split(pat=',', n=1).str
data['latitude'] = data['latitude'].astype(float)
data['longitude'] = data['longitude'].astype(float)

# Load the GeoJSON file
geojson_path = 'data/cb_2018_us_div.geojson'
geojson_data = gpd.read_file(geojson_path)

# Function to get the region name based on latitude and longitude
def get_region_name(lat, lon):
    point = Point(lon, lat)
    for index, row in geojson_data.iterrows():
        if row['geometry'].contains(point):
            return row['NAME']
    return None

# Add region column
data['region'] = data.apply(lambda row: get_region_name(row['latitude'], row['longitude']), axis=1)

# Sample 100 rows per region (or state if you modify accordingly)
subset_data = data.groupby('region').apply(lambda x: x.sample(n=500, replace=True)).reset_index(drop=True)

# Save the processed CSV
subset_data.to_csv('data/processed/FM_service_contour_current_processed.csv', index=False)

print("done.")

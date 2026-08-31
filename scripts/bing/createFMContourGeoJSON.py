import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

# Load the data from the CSV file
csv_file = 'data/raw/FM_service_contour_NYS.csv'
data = pd.read_csv(csv_file, dtype=str)

# Define the column that contains the point coordinates
point_column = 'transmitter_site'  # Adjust if needed

# Functions to create geometries
def create_point(row):
    try:
        x, y = map(float, row[point_column].strip('"').split(','))
        return Point(y, x)
    except (ValueError, TypeError):
        return None

def create_linestring(row):
    try:
        coordinates = [(float(row[str(i)].split(',')[1]), float(row[str(i)].split(',')[0]))
                       for i in range(360)]
        # Ensure the LineString is closed by adding the first point at the end
        coordinates.append(coordinates[0])
        return LineString(coordinates)
    except (ValueError, TypeError):
        return None

def create_polygon(row):
    try:
        coordinates = [(float(row[str(i)].split(',')[1]), float(row[str(i)].split(',')[0]))
                       for i in range(360)] + [(float(row['0'].split(',')[1]), float(row['0'].split(',')[0]))]
        return Polygon(coordinates)
    except (ValueError, TypeError):
        return None

# Create a GeoDataFrame for each geometry type
for feature_type in ['Point', 'LineString', 'Polygon']:
    gdf = gpd.GeoDataFrame(data)

    # Create the 'geometry' column based on the feature type
    if feature_type == 'Point':
        gdf['geometry'] = gdf.apply(create_point, axis=1)
    elif feature_type == 'LineString':
        gdf['geometry'] = gdf.apply(create_linestring, axis=1)
    elif feature_type == 'Polygon':
        gdf['geometry'] = gdf.apply(create_polygon, axis=1)

    # Drop rows with invalid geometries
    gdf = gdf.dropna(subset=['geometry'])

    # Drop the coordinate columns as they are now encoded in the geometry
    if feature_type != 'Point':
        gdf.drop(columns=[str(i) for i in range(360)], inplace=True)

    # Ensure CRS is set to WGS 84
    gdf.set_crs(epsg=4326, inplace=True)

    # Save the GeoDataFrame as a GeoJSON
    geojson_file = f'data/processed/FM_contours_NYS_{feature_type}.geojson'
    gdf.to_file(geojson_file, driver='GeoJSON')

    print(f"{feature_type} GeoJSON file created: {geojson_file}")

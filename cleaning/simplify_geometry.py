import geopandas as gpd
from tqdm import tqdm

def simplify_geojson(input_file, output_file, tolerance=0.01):
    """
    Simplify the geometry of a GeoJSON file using the Douglas-Peucker algorithm.

    :param input_file: Path to the input GeoJSON file.
    :param output_file: Path where the simplified GeoJSON will be saved.
    :param tolerance: Tolerance parameter for the simplification. Higher values mean more simplification.
    """
    
    # Read the GeoDataFrame from the input file
    gdf = gpd.read_file(input_file)
    
    # Get the number of features for the progress bar
    total_features = len(gdf)
    
    # Create a tqdm progress bar with ASCII format
    pbar = tqdm(total=total_features, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}', ascii=True, desc="Simplifying")
    
    # Simplify the geometry with tqdm updates
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: _simplify_with_progress(geom, tolerance, pbar))
    
    # Close the progress bar
    pbar.close()

    # Save the simplified GeoDataFrame to the output file
    gdf.to_file(output_file, driver="GeoJSON")

def _simplify_with_progress(geom, tolerance, pbar):
    """
    Helper function to simplify a geometry and update the tqdm progress bar.
    """
    simplified_geom = geom.simplify(tolerance, preserve_topology=True)
    pbar.update(1)
    return simplified_geom

if __name__ == "__main__":
    input_geojson = "data/processed/contour_maps/sliced_USA_contours.geojson"
    output_geojson = "data/processed/contour_maps/sliced_USA_contours_simplified.geojson"
    
    simplify_geojson(input_geojson, output_geojson)
    print('done.')

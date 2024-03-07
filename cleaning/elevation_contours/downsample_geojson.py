import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import LineString, MultiLineString

def downsample_geojson(input_file, output_file, downsample_factor=10):
    """
    Downsample the coordinates of a GeoJSON file.

    :param input_file: Path to the input GeoJSON file.
    :param output_file: Path where the downsampled GeoJSON will be saved.
    :param downsample_factor: Factor by which coordinates will be downsampled. e.g. 10 means every 10th point is taken.
    """
    
    # Read the GeoDataFrame from the input file
    gdf = gpd.read_file(input_file)
    
    # Get the number of features for the progress bar
    total_features = len(gdf)
    
    # Create a tqdm progress bar with ASCII format
    pbar = tqdm(total=total_features, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}', ascii=True, desc="Downsampling")
    
    # Downsample the geometry with tqdm updates
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: _downsample_with_progress(geom, downsample_factor, pbar))
    
    # Close the progress bar
    pbar.close()

    # Save the downsampled GeoDataFrame to the output file
    gdf.to_file(output_file, driver="GeoJSON")

def _downsample_with_progress(geom, factor, pbar):
    """
    Helper function to downsample a geometry and update the tqdm progress bar.
    """
    if isinstance(geom, LineString):
        coords = geom.coords[::factor]
        if len(coords) < 2:
            coords = geom.coords[:2]  # At least keep the start and end points
        downsampled_geom = LineString(coords)
    elif isinstance(geom, MultiLineString):
        lines = []
        for line in geom.geoms:
            coords = line.coords[::factor]
            if len(coords) >= 2:
                lines.append(LineString(coords))
        downsampled_geom = MultiLineString(lines)
    else:
        downsampled_geom = geom
    pbar.update(1)
    return downsampled_geom


if __name__ == "__main__":
    input_geojson = "/Users/matthewheaton/Documents/GitHub/imagery_scraper/data/raw/NYS_elevContour_SimplifyLine_1500mCurve_downsample30_20231120_low_detail.geojson"
    output_geojson = "data/processed/contour_maps/simplified/NYS_elevContour_SimplifyLine_1500mCurve_downsampleExtra_20231120_low_detail.geojson"
    
    downsample_geojson(input_geojson, output_geojson)
    print('done.')

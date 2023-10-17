import ijson
import json
from tqdm import tqdm
from decimal import Decimal

def decimal_default(obj):
    """
    Handle Decimal types during JSON serialization.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def slice_large_geojson_by_size(input_file, output_file, max_size_mb=10):
    """
    Slices a large GeoJSON file by reading it in chunks and writing a subset based on file size.

    :param input_file: Path to the input GeoJSON file.
    :param output_file: Path where the sliced GeoJSON will be saved.
    :param max_size_mb: Desired maximum size of the output file in MB.
    """
    MAX_SIZE = max_size_mb * 1024 * 1024

    with open(input_file, 'rb') as fin, open(output_file, 'w') as fout:
        fout.write('{ "type": "FeatureCollection", "features": [')
        
        parser = ijson.items(fin, 'features.item')
        written = False
        for feature in tqdm(parser, desc="Processing features"):
            # Convert feature dictionary to a JSON string
            feature_str = json.dumps(feature, default=decimal_default)
            feature_str = f",\n{feature_str}" if written else f"\n{feature_str}"
            
            if fout.tell() + len(feature_str) > MAX_SIZE:
                break
            fout.write(feature_str)
            written = True
        
        fout.write("\n]}")

if __name__ == "__main__":
    input_geojson = "data/raw/USA_contours_stanford.json"
    output_geojson = "data/processed/contour_maps/sliced_USA_contours.geojson"
    
    slice_large_geojson_by_size(input_geojson, output_geojson)
    print('done.')

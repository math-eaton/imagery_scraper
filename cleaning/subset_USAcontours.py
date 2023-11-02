import ijson
import json
import os
from tqdm import tqdm
from decimal import Decimal

def decimal_default(obj):
    """
    Handle Decimal types during JSON serialization.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def slice_large_geojson_by_size(input_file, output_dir, base_filename, max_size_mb=50):
    """
    Slices a large GeoJSON file by reading it in chunks and writing multiple subset files based on file size.

    :param input_file: Path to the input GeoJSON file.
    :param output_dir: Directory where the subset files will be saved.
    :param base_filename: Base name for output subset files.
    :param max_size_mb: Desired maximum size of each subset file in MB.
    """
    MAX_SIZE = max_size_mb * 1024 * 1024

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_count = 1
    fout = open(os.path.join(output_dir, f"{base_filename}_{file_count}.geojson"), 'w')
    fout.write('{ "type": "FeatureCollection", "features": [')
    
    first_feature_in_file = True

    with open(input_file, 'rb') as fin:
        parser = ijson.items(fin, 'features.item')
        for feature in tqdm(parser, desc="Processing features"):
            # Convert feature dictionary to a JSON string
            feature_str = json.dumps(feature, default=decimal_default)
            
            # If adding the current feature surpasses the MAX_SIZE, close the current file and open a new one
            if fout.tell() + len(feature_str) + len(",\n]}") > MAX_SIZE:
                fout.write("\n]}")
                fout.close()
                file_count += 1
                fout = open(os.path.join(output_dir, f"{base_filename}_{file_count}.geojson"), 'w')
                fout.write('{ "type": "FeatureCollection", "features": [')
                first_feature_in_file = True
            else:
                if not first_feature_in_file:
                    fout.write(",")  # Write comma before the next feature

            fout.write(feature_str)

            if first_feature_in_file:
                first_feature_in_file = False

        # Close the last file
        fout.write("\n]}")
        fout.close()

if __name__ == "__main__":
    input_geojson = "data/raw/USA_contours_stanford.json"
    output_dir = "data/processed/contour_maps/subset"
    base_filename = "sliced_USA_contours"
    
    slice_large_geojson_by_size(input_geojson, output_dir, base_filename)
    print('done.')

import geojson

def process_geojson(file_path, stride):
    with open(file_path, 'r') as file:
        data = geojson.load(file)

    new_features = []
    for feature in data['features']:
        if feature['geometry']['type'] == 'LineString':
            new_coords = feature['geometry']['coordinates'][::stride]
            new_feature = geojson.Feature(geometry=geojson.LineString(new_coords))
            new_features.append(new_feature)
        else:
            # Add non-LineString features as-is
            new_features.append(feature)

    return geojson.FeatureCollection(new_features)

def save_geojson(data, file_path):
    with open(file_path, 'w') as file:
        geojson.dump(data, file)

# Paths to the input and output files
input_file_path = 'data/processed/contour_maps/simplified/NYS_elevContours_DEM_simplify750curve_20231119_simplified.geojson'

# Process and save for each LOD
for stride, detail in zip([100, 60, 20], ['low', 'medium', 'high']):
    output_file_path = f'data/processed/contour_maps/simplified/NYS_elevContours_DEM_simplify750curve_20231119_simplified_{detail}_detail.geojson'
    processed_data = process_geojson(input_file_path, stride)
    save_geojson(processed_data, output_file_path)

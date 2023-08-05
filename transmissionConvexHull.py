import pandas as pd
import folium

# Create a Map instance
m = folium.Map(location=[45.5236, -122.6750], zoom_start=2)  # You can adjust the location and zoom level

# Specify the chunk size for reading the CSV
chunksize = 1000

# Load the data in chunks
for df_chunk in pd.read_csv('data/FM_service_contour_current.csv', chunksize=chunksize, nrows=5000):
    for _, row in df_chunk.iterrows():
        # get columns named '0', '1', ..., '359' and drop missing values
        coordinates = row[[str(i) for i in range(360)]].dropna()
        # convert strings "latitude, longitude" into list of [latitude, longitude]
        coordinates = [coord.split(',') for coord in coordinates]
        # transpose list of coordinates into two lists: latitudes and longitudes
        latitudes, longitudes = map(list, zip(*((float(coord[0]), float(coord[1])) for coord in coordinates)))
        # Create a list of coordinates
        locations = list(zip(latitudes, longitudes))
        # Create a PolyLine:
        folium.PolyLine(locations=locations, color='blue', weight=2.5, opacity=1).add_to(m)

# Save it as html
m.save('output/folium/output.html')  # replace with your file path

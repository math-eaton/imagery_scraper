import pandas as pd
import folium

dtype = {
    'application_id': str,
    'service': str,
    'lms_application_id': str,
    'dts_site_number': str,
}

# add latitude and longitude fields to dtype
for i in range(1, 362):
    dtype[f'latitude_{i}'] = float
    dtype[f'longitude_{i}'] = float

# read only necessary columns
necessary_columns = list(dtype.keys())
df = pd.read_csv('data/FM_service_contour_processed.csv', delimiter=',', dtype=dtype, usecols=necessary_columns)

df['application_id'] = pd.to_numeric(df['application_id'], errors='coerce')
df['dts_site_number'] = pd.to_numeric(df['dts_site_number'], errors='coerce')

# map instance
m = folium.Map(location=[45.5236, -122.6750], zoom_start=2)

for _, row in df.iterrows():
    latitudes = row[[f'latitude_{i}' for i in range(1,362)]].values
    longitudes = row[[f'longitude_{i}' for i in range(1,362)]].values
    locations = list(zip(latitudes, longitudes))
    folium.PolyLine(locations=locations, color='blue', weight=2.5, opacity=1).add_to(m)

m.save('output/folium/output.html') 

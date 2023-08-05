import pandas as pd

# Read the csv file
df = pd.read_csv('data/FM_service_contour_processed.csv')

# Unpivot/melt the dataframe from wide to long format separately for latitude and longitude
df_lat = df.melt(id_vars=['application_id', 'service', 'lms_application_id', 'dts_site_number', 'transmitter_site'], 
                 value_vars=[f'latitude{i}' for i in range(1, 362)], 
                 var_name='latitude_point', 
                 value_name='latitude')

df_long = df.melt(id_vars=['application_id', 'service', 'lms_application_id', 'dts_site_number', 'transmitter_site'], 
                  value_vars=[f'longitude{i}' for i in range(1, 362)], 
                  var_name='longitude_point', 
                  value_name='longitude')

# Create point index to merge latitude and longitude dataframes
df_lat['point_index'] = df_lat['latitude_point'].str.extract('(\d+)')
df_long['point_index'] = df_long['longitude_point'].str.extract('(\d+)')

# Merge two dataframes
df_final = pd.merge(df_lat, df_long, on=['application_id', 'service', 'lms_application_id', 'dts_site_number', 'transmitter_site', 'point_index'])

# Drop unnecessary columns
df_final.drop(['latitude_point', 'longitude_point'], axis=1, inplace=True)

# Drop rows where either latitude or longitude is NaN
df_final.dropna(subset=['latitude', 'longitude'], inplace=True)

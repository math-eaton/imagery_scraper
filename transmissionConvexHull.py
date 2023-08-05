import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv('data/fm_contours_sample.csv') # replace with your file path

plt.figure(figsize=(10,10))

# loop through rows and plot each journey as a separate line
for _, row in df.iterrows():
    # get columns named '0', '1', ..., '359' and drop missing values
    coordinates = row[[str(i) for i in range(360)]].dropna()
    # convert strings "latitude, longitude" into list of [latitude, longitude]
    coordinates = [coord.split(',') for coord in coordinates]
    # transpose list of coordinates into two lists: latitudes and longitudes
    latitudes, longitudes = map(list, zip(*((float(coord[0]), float(coord[1])) for coord in coordinates)))
    plt.plot(longitudes, latitudes)

plt.show()

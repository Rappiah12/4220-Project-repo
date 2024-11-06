import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import folium
from folium import plugins




# Check in data loading into dataframe
checkins = pd.read_csv(
    'Gowalla_totalCheckins.txt',
    sep=r'\s+',  
    names=['user', 'checkin_time', 'latitude', 'longitude', 'location_id'],
    encoding='utf-8'
)

# Converting check in time for easier handling 
checkins['checkin_time'] = pd.to_datetime(checkins['checkin_time'], errors='coerce')

# TEST FOR PRINTING FIRST FEW ROWS (can be deleted later) 
# print(checkins.head())



#
#
#
### At this point, the check in data has been loaded in correctly ### 
#
#
#



# Defining latitude and longitude bounds for Aurora, CO
lat_min, lat_max = 39.6, 39.8
lon_min, lon_max = -104.9, -104.7

# Filtering check-ins made only in Aurora
aurora_checkins = checkins[
    (checkins['latitude'] >= lat_min) & (checkins['latitude'] <= lat_max) &
    (checkins['longitude'] >= lon_min) & (checkins['longitude'] <= lon_max)
].copy()

# Export filtered data to CSV file 
# aurora_checkins.to_csv('aurora_checkins.csv', index=False)
# print("\nFiltered data for Aurora has been saved to 'aurora_checkins.csv'.")



#
#
#
### We now have separated the data into only those for Aurora ### 
#
#
#



# Extracting latitude and longitude so we can cluster
coords = aurora_checkins[['latitude', 'longitude']].values

# We are going to be using KMeans to create 15 clusters of our
# data in order to identify and plot them
kmeans = KMeans(n_clusters=15, random_state=0).fit(coords)
aurora_checkins['cluster'] = kmeans.labels_


""" # We will then plot this data, using latitude/longitude as our axes
# and each cluster having their own color
plt.figure(figsize=(10, 8))
plt.scatter(aurora_checkins['longitude'], aurora_checkins['latitude'],
            c=aurora_checkins['cluster'], cmap='viridis', marker='o', s=10, alpha=0.6)

plt.title("Aurora Check-ins Clustered by K-Means (15 Clusters)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label="Cluster ID")
plt.grid(True)

plt.show() """


# We are going to use Folium to plot our check-ins and visualize them
# on a satellite image (interactive)
map_center = [39.7294, -104.8319]  # Center of Aurora (approximate)
m = folium.Map(
    location=map_center,
    zoom_start=12,
    tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attr="Map data © OpenStreetMap contributors"
)

# Plot each check in as a circle 
for _, row in aurora_checkins.iterrows():
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=4,
        color=f'#{row["cluster"] * 123456 % 0xFFFFFF:06x}', # Color based on check-in's associated cluster
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

# Save the map to an HTML file without additional layers
m.save("aurora_checkins_map.html")

print("Map saved as 'aurora_checkins_map.html'. Open it in a web browser to view the map.")


#
#
#
### Our data is now clustered into 15 groups and plotted ### 
#
#
#





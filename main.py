import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import folium
from folium import plugins
import networkx as nx



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

# Save the map to an HTML file 
m.save("aurora_checkins_map.html")

print("Map saved as 'aurora_checkins_map.html'. Open it in a web browser to view the map.")


#
#
#
### Our data is now clustered into 15 groups and plotted ### 
#
#
#



# This will group users by the clusters they checked into
user_clusters = aurora_checkins.groupby('cluster')['user'].apply(set).to_dict()


# Initialize graph
G = nx.Graph()

# This will add edges between users who visited the same cluster
for cluster, users in user_clusters.items():
    users = list(users)  
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            if G.has_edge(users[i], users[j]):
                # Increase weight if edge already exist
                G[users[i]][users[j]]['weight'] += 1
            else:
                # Create edge with weight of 1
                G.add_edge(users[i], users[j], weight=1)


print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")


# Calculate/Print Network Properties
print(f"Network density: {nx.density(G)}")
print(f"Average clustering coefficient: {nx.average_clustering(G)}")

# Degree centrality
degree_centrality = nx.degree_centrality(G)

# Creates a list of top users by centrality, helpful in showing key players 
# in the network
top_users = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"Top 5 users by degree centrality: {top_users}")

# Print network graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title("User Location Network")
plt.show()



#
#
#
### Our network graph with edges is now complete, showing graph properties 
#
#
#

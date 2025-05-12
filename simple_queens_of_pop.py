# Lily Gates
# Sun. 5/11/25

import yaml  # Save keys
import numpy as np 
import pandas as pd 
import spotipy  # For Spotify API

from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import time  # for buffering, sleep function

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

##############################################
##### IMPORT CREDENTIALS FROM 'keys.yml' #####
##############################################

# Load credentials from 'keys.yml'
with open("keys.yml", 'r') as file:
    keys = yaml.safe_load(file)

CLIENT_ID = keys['spotify_song_key']
CLIENT_SECRET = keys['spotify_song_pass']
REDIRECT_URI = keys['spotify_uri']
SCOPE = "user-library-read"

#####################################
##### SPOTIFY API AUTHENTICATION ####
#####################################

# Set environment variables or replace with your credentials

# Initialize Spotify API client
sp = Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                       client_secret=CLIENT_SECRET,
                                       redirect_uri=REDIRECT_URI,
                                       scope=SCOPE))


#########################################
##### SEARCH FOR ARTISTS & THEIR ID's ####
#########################################

# List of artist names from Billboard's Top 100 Women Artists of the 21st Century
artist_names = [
    "Taylor Swift", "Rihanna", "Beyoncé", "Adele", "Katy Perry", "Lady Gaga", "P!nk", "Ariana Grande",
    "Miley Cyrus", "Alicia Keys", "Kelly Clarkson", "Mariah Carey", "Carrie Underwood", "Britney Spears",
    "Billie Eilish", "Nicki Minaj", "Destiny’s Child", "SZA", "Avril Lavigne", "Christina Aguilera",
    "Olivia Rodrigo", "Jennifer Lopez", "Mary J. Blige", "Doja Cat", "Dua Lipa", "Kesha", "Cardi B", "Fergie",
    "Ashanti", "Selena Gomez", "Norah Jones", "Gwen Stefani", "Meghan Trainor", "The Chicks", "Faith Hill",
    "Madonna", "Halsey", "Ciara", "Evanescence", "Missy Elliott", "Janet Jackson", "Lizzo", "Nelly Furtado",
    "Lorde", "Shakira", "Sugarland", "Aaliyah", "Celine Dion", "The Pussycat Dolls", "Sabrina Carpenter",
    "Camila Cabello", "Demi Lovato", "Sheryl Crow", "Colbie Caillat", "Sia", "Miranda Lambert", "Leona Lewis",
    "Keyshia Cole", "No Doubt", "Dido", "Enya", "Ellie Goulding", "Megan Thee Stallion", "Sara Bareilles",
    "Jessica Simpson", "Shania Twain", "Eve", "Natasha Bedingfield", "Whitney Houston", "Iggy Azalea",
    "Lana Del Rey", "Amy Winehouse", "Hilary Duff", "Gretchen Wilson", "Carly Rae Jepsen", "Susan Boyle",
    "Jordin Sparks", "Ashlee Simpson", "Martina McBride", "Chappell Roan", "Fleetwood Mac", "The Band Perry",
    "Michelle Branch", "Alessia Cara", "Summer Walker", "Reba McEntire", "Keri Hilson", "Barbra Streisand",
    "Myá", "Bebe Rexha", "Paramore", "Lil’ Kim", "Brenda Lee", "Toni Braxton", "Charli XCX", "Sade",
    "Fifth Harmony", "Fantasia", "Vanessa Carlton", "Danity Kane"
]

# Corresponding rankings (this list should be in the same order as artist_names)
rankings = list(range(1, 101))  # Assuming the artists are ranked from 1 to 100

# Initialize an empty list to store artist IDs
artist_ids = []

# Retrieve artist IDs by searching for artist names
for artist_name in artist_names:
    try:
        result = sp.search(q=artist_name, type="artist", limit=1)  # Limit to 1 result per artist
        artist_id = result['artists']['items'][0]['id']  # Get the artist ID
        artist_ids.append(artist_id)
        time.sleep(0.5)  # Sleep for 0.5 seconds between requests to avoid hitting rate limit
    except Exception as e:
        print(f"Error fetching data for {artist_name}: {e}")
        time.sleep(1)  # Increase sleep time if there's an error, e.g., rate limit exceeded

# Collect artist metadata
artist_data = []
for idx, artist_id in enumerate(artist_ids):
    try:
        artist = sp.artist(artist_id)
        artist_data.append({
            "ranking": rankings[idx],  # Get the artist's ranking
            "name": artist["name"],
            "id": artist["id"],
            "genres": artist["genres"],
            "popularity": artist["popularity"],
            "followers": artist["followers"]["total"]
        })
        time.sleep(0.5)  # Buffer time between each artist data retrieval
    except Exception as e:
        print(f"Error fetching metadata for {artist_id}: {e}")
        time.sleep(1)  # Increase sleep time if there's an error

# Create DataFrame
df = pd.DataFrame(artist_data)

################################
##### CLEANING DATA & ##########
###### REFORMATTING DATAFRAME ##
################################

# One-hot encoding for genres
df_genres = df["genres"].explode().str.get_dummies()

# Group by artist to ensure each genre column has 1 if the artist belongs to that genre
df_genres = df_genres.groupby(df_genres.index).max()

# Combine the relevant features: ranking, popularity, followers, and the one-hot encoded genres
df_combined = pd.concat([df[["ranking", "popularity", "followers"]], df_genres], axis=1)

# Normalize numerical features (popularity, followers)
df_combined[["ranking", "popularity", "followers"]] = df_combined[["ranking", "popularity", "followers"]].apply(lambda x: (x - x.mean()) / x.std())

# Now df_combined is ready for KMeans clustering
#print(df_combined.head())  # Check the cleaned DataFrame

##############################
##### CALCULATING THE    #####
##### ELBOW METHOD FOR K #####
##############################

# Elbow Method for finding the best number of clusters (K)
inertia_values = []
start = 1  # Start K at 1
stop = 31  # Final K = 30


# Calculate inertia for K values from 1 to 30
for k in range(start, stop):
    kmeans = KMeans(n_clusters=k, n_init=10)  # n_init set to 10 for robustness
    kmeans.fit(df_combined)  # Use df_combined for clustering (excluding name/id)
    
    inertia_values.append(kmeans.inertia_)  # Store the inertia value for each K

# Print the inertia values to observe
for i, inertia in enumerate(inertia_values):
    print(f"K={i+1}: Inertia={inertia}")

###########################
##### PLOTTING THE    #####
##### ELBOW METHOD FOR K ###
############################

# Plotting the Elbow Method
plt.figure(figsize=(15, 6))
plt.title("KMeans Elbow Method", fontsize=16, fontweight='bold')
plt.xlabel("Number of Clusters (K)", fontsize=14)
plt.ylabel("WCSS (Within-cluster sum of squares)", fontsize=14)

# Plot the regular inertia curve
plt.plot(range(1, 31), inertia_values, marker='o', color='b', linestyle='-', markersize=8, linewidth=2, label="Inertia")

# Highlight the "elbow" point (choose the optimal K based on visual analysis)
optimal_k = 10  # This will depend on visual analysis
plt.scatter(optimal_k, inertia_values[optimal_k - 1], color='r', s=100, zorder=5, label=f"Optimal K = {optimal_k}")

# Customize Grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add a legend
plt.legend()

# Customize x-ticks
plt.xticks(range(1, 31), fontsize=12)
plt.yticks(fontsize=12)

# Show the graph
plt.show()

############################
##### CREATING CLUSTERS ####
############################

# Apply KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)

# Fit the model to your data
kmeans.fit(df_combined)

# Get the cluster assignments for each data point
df_combined['Cluster'] = kmeans.labels_

# Store the cluster centers for each cluster
cluster_centers = kmeans.cluster_centers_

# List all columns in the dataframe
#print(df_combined.columns)

###########################
##### ARTISTS IN EA   #####
##### CLUSTER, SORTED #####
###########################

print("\n" + "-"*50 + "\n")
print("\n" + "-"*50 + "\n")

# Add the cluster labels to the original df
df['Cluster'] = df_combined['Cluster']

# Filter DataFrame for each cluster
for cluster_num in range(12):
    # Filter the DataFrame for each cluster and reset the index
    filtered_df = df[df['Cluster'] == cluster_num].reset_index(drop=True)
    
    # Select only the 'artist' and 'ranking' columns
    cluster_info = filtered_df[['name', 'ranking']]

    # Display the filtered DataFrame for the current cluster
    print(f"Cluster {cluster_num} Data:")
    print(cluster_info)
    print("\n" + "-"*50 + "\n")

################################
##### CLUSTER FEATURES      ####
##### AVERAGES & TOP GENRES ####
################################

# Sum up the one-hot encoded genre columns for each cluster in df_combined
cluster_genres = df_combined.groupby('Cluster')[df_genres.columns].sum()

# Ensure the columns are numeric (if not already)
cluster_genres = cluster_genres.apply(pd.to_numeric, errors='coerce')

# Get the top 5 genres for each cluster (genres with the highest sums)
top_genres = cluster_genres.apply(lambda x: x.nlargest(5).index.tolist(), axis=1)

# Calculate average values for ranking, followers, and popularity per cluster
cluster_averages = df_combined.groupby('Cluster')[['ranking', 'popularity', 'followers']].mean()

# Combine the average values with the top genres
cluster_summary = pd.concat([cluster_averages, top_genres], axis=1)
cluster_summary.columns = ['Avg Ranking', 'Avg Followers', 'Avg Popularity', 'Top 5 Genres']

# Convert Clusters Summary into .CSV
cluster_summary.to_csv('cluster_summary.csv')

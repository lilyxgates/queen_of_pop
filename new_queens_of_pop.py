# Lily Gates
# May 2025

import os  # To get current directory and save results
import json  # To save cache info to prevent re-running API requests unecessarily
import yaml  # Read saved keys
import numpy as np 
import pandas as pd 

import requests
from bs4 import BeautifulSoup  # For scraping Wikipedia for genres

import spotipy  # For Spotify API
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import time  # For buffering, sleep function

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

import difflib # For searching for similar names
import unicodedata # Remove accent marks and normalize Unicode characters
import re  # Formatting symbols

from collections import defaultdict


##############################################
##### IMPORT CREDENTIALS FROM 'keys.yml' #####
##############################################

# Load credentials from 'keys.yml'
with open("keys.yml", 'r') as file:
    keys = yaml.safe_load(file)

CLIENT_ID = keys['spotify_popqueen_key']
CLIENT_SECRET = keys['spotify_popqueen_pass']
REDIRECT_URI = keys['spotify_uri']
SCOPE = "user-library-read"


#########################################
##### SEARCH FOR ARTISTS & THEIR ID's ###
#########################################

# Normalize function
def normalize(text):
    if not text:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()

# -------------------------------
# Close match check using difflib
def is_close_match(input_name, spotify_name, cutoff=0.8):
    norm_input = normalize(input_name)
    norm_spotify = normalize(spotify_name)
    return difflib.SequenceMatcher(None, norm_input, norm_spotify).ratio() >= cutoff

# -------------------------------
# Function to fetch genres - with flagging when artists have no genre
def fetch_genres(sp, artist_id, artist_name):
    try:
        artist = sp.artist(artist_id)
        genres = artist.get('genres', [])
        if genres:
            return genres
        else:
            print(f"No genres found for artist {artist_name}.")
            return []
    except spotipy.SpotifyException as e:
        print(f"Error fetching genres for artist {artist_name}: {e}")
        return []


def fetch_artist_metadata(artist_name, sp):
    try:
        # Check if artist data exists in cache
        if artist_name in cache:
            print(f"Cache hit for {artist_name}")
            artist_data = cache[artist_name]

            # If genres are missing, fetch them
            if not artist_data.get('genres'):
                print(f"Genres missing for {artist_name}, fetching...")
                genres = fetch_genres(sp, artist_data["id"], artist_name)
                artist_data["genres"] = genres  # Update genres in cache
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cache, f, indent=4)
        
            return artist_data
        
        # Search for artist in Spotify
        result = sp.search(q=artist_name, type="artist", limit=5)
        best_match = None

        for artist in result['artists']['items']:
            if is_close_match(artist_name, artist['name']):
                best_match = artist
                break
        
        if best_match:
            artist_id = best_match['id']
            genres = fetch_genres(sp, artist_id, artist_name)
            popularity = best_match['popularity']
            followers = best_match['followers']['total']
            
            metadata = {
                "id": artist_id,
                "name": best_match['name'],
                "genres": genres,
                "popularity": popularity,
                "followers": followers
            }

            # Save the metadata in cache
            cache[artist_name] = metadata

            # Save updated cache to file
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=4)

            return metadata
        else:
            print(f"WARNING: No good match for {artist_name}")
            return None
    except Exception as e:
        print(f"Error fetching metadata for {artist_name}: {e}")
        return None

#####################################
##### SPOTIFY API AUTHENTICATION ####
#####################################

# Initialize Spotify API client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                                client_secret=CLIENT_SECRET,
                                                redirect_uri=REDIRECT_URI,
                                                scope=SCOPE))



#########################################
##### SEARCH FOR ARTISTS & THEIR ID's ###
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

# Assign rankings starting from 1
rankings = list(range(1, len(artist_names) + 1))

# Search and collect artist metadata (genres, popularity, followers)
artist_data = []
rankings = list(range(1, len(artist_names) + 1))
artist_data = []
artists_no_genre = []  # List to track artists with no genre

for i, artist_name in enumerate(artist_names):
    metadata = fetch_artist_metadata(artist_name, sp)
    
    if metadata:
        # If genres are missing, try to refresh using sp.artist()
        if not metadata["genres"]:
            print(f"Refetching genres for {metadata['name']}...")
            genres = fetch_genres(sp, metadata["id"], metadata["name"])  # Pass artist_name here
            metadata["genres"] = genres or []  # Fallback to empty list if still no genres
            # Update cache with new genres
            cache[artist_name] = metadata
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=4)

        metadata["ranking"] = rankings[i]
        artist_data.append(metadata)
        
        # Track artists with no genres
        if not metadata["genres"]:
            artists_no_genre.append(artist_name)

    time.sleep(0.5)

# Convert the artists_no_genre list into a DataFrame
artists_no_genre_df = pd.DataFrame(artists_no_genre, columns=['Artist Name'])
print("\nArtists with no genres:")
print(artists_no_genre_df)

# Create DataFrame with results
artist_df = pd.DataFrame(artist_data)

# Show the final result
print("\nSUCCESS: Artist search complete!\n")
print(artist_df)

# Save to CSV
artist_df_file = "spotify_artist_metadata.csv"
current_directory = os.getcwd()
artist_df_file_path = os.path.join(current_directory, artist_df_file)
artist_df.to_csv(artist_df_file_path, index=False)
print(f"\nA .csv file has been saved:\n'{artist_df_file_path}'\n")

print(artist_df.columns)


###################################
##### MERGE GENRES FROM ###########
##### BOTH WIKI AND SPOTIFY #######
###################################

################################
##### CLEANING DATA & ##########
###### REFORMATTING DATAFRAME ##
################################

# One-hot encoding for genres
df_genres = artist_df["genres"].explode().str.get_dummies()

# Group by artist to ensure each genre column has 1 if the artist belongs to that genre
df_genres = df_genres.groupby(df_genres.index).max()

# Combine the relevant features: ranking, popularity, followers, and the one-hot encoded genres
df_combined = pd.concat([artist_df[["ranking", "popularity", "followers"]], df_genres], axis=1)

# Normalize numerical features (popularity, followers)
df_combined[["ranking", "popularity", "followers"]] = df_combined[["ranking", "popularity", "followers"]].apply(lambda x: (x - x.mean()) / x.std())

# Now df_combined is ready for KMeans clustering
#print(df_combined.head())  # Check the cleaned DataFrame

# Save to CSV (optional)
df_combined_file = "spotify_artist_metadata_genre_onehot_encoded.csv"
current_directory = os.getcwd()
df_combined_file_path = os.path.join(current_directory, df_combined_file)

# Save the DataFrame to the specified file path
df_combined.to_csv(df_combined_file_path, index=False)
print(f"\nA .csv file has been saved:\n'{df_combined_file_path}'\n")



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

# Spacer
print("\n" + "-"*50 + "\n")

# Print the inertia values to observe
for i, inertia in enumerate(inertia_values):
    print(f"K={i+1}: Inertia={inertia:.4f}")  # Round inertia value to 4 decimals

# Spacer
print("\n" + "-"*50 + "\n")


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

# Highlight the "elbow" point (choose the optimal K based on your analysis)
optimal_k = 10
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

# Spacer
print("\n" + "-"*50 + "\n")

# Add the cluster labels to the original df
artist_df['Cluster'] = df_combined['Cluster']

# Filter DataFrame for each cluster using 'optimal_k' from elbow method
for cluster_num in range(optimal_k):
    # Filter the DataFrame for each cluster and reset the index
    filtered_df = artist_df[artist_df['Cluster'] == cluster_num].reset_index(drop=True)
    
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

# Rename the columns in cluster_averages
cluster_averages.columns = ['Avg Ranking', 'Avg Popularity', 'Avg Followers']

# Melt the dataframe to reshape it for seaborn
melted_averages = cluster_averages.reset_index().melt(
    id_vars="Cluster", 
    value_vars=["Avg Ranking", "Avg Popularity", "Avg Followers"],
    var_name="Feature", 
    value_name="Value"
)

# Plotting cluster average features
plt.figure(figsize=(14, 6))
sns.barplot(data=melted_averages, x="Cluster", y="Value", hue="Feature", palette="Set2")
plt.title("Average Feature Values by Cluster", fontsize=16, fontweight='bold')
plt.xlabel("Cluster", fontsize=14)
plt.ylabel("Standardized Value", fontsize=14)
plt.legend(title="Feature", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



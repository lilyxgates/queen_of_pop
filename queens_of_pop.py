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

#####################################
##### SPOTIFY API AUTHENTICATION ####
#####################################

# Initialize Spotify API client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                                client_secret=CLIENT_SECRET,
                                                redirect_uri=REDIRECT_URI,
                                                scope=SCOPE))

###############################
##### CREATE AND SAVE CACHE ###
###############################

# Merged genres for standardization
merged_genres = {
    'pop': 'Pop',
    'bubblegum pop': 'Pop',
    'soft pop': 'Pop',
    'art pop': 'Pop',
    'classic rock': 'Rock',
    'rock': 'Rock',
    'emo': 'Rock',
    'emo pop': 'Rock',
    'pop punk': 'Rock',
    'alternative metal': 'Rock',
    'r&b': 'R&B',
    'neo soul': 'R&B',
    'soul': 'R&B',
    'hip hop': 'Hip Hop',
    'rap': 'Hip Hop',
    'hip-hop': 'Hip Hop',
    'dance': 'Electronic',
    'electronic': 'Electronic',
    'new age': 'Electronic',
    'house': 'Electronic',
    'techno': 'Electronic',
    'disco': 'Electronic',
    'hyperpop': 'Electronic',
    'country': 'Country',
    'latin pop': 'Latin',
    'latin': 'Latin',
    'folk': 'Folk',
    'celtic': 'Folk',
    'jazz': 'Standards/Traditional',
    'blues': 'Standards/Traditional',
    'adult standards': 'Standards/Traditional',
    'metal': 'Rock',
    'punk': 'Rock',
    'alternative': 'Rock',
    'indie': 'Pop',
    'funk': 'R&B',
    'dancehall': 'Hip Hop',
    'grunge': 'Rock',
    'new jack swing': 'R&B',
    'christmas': 'Seasonal',
    'soft rock': 'Rock',
    'variété française': 'International',
    'east coast hip hop': 'Hip Hop'
}

# Cache to store previously scraped genres
wiki_genre_cache = {}

# Genre scraping function with caching
def get_genres_from_wikipedia(artist_name):
    if artist_name in wiki_genre_cache:
        return wiki_genre_cache[artist_name]

    search_url = f"https://en.wikipedia.org/wiki/{artist_name.replace(' ', '_')}"
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            infobox = soup.find("table", {"class": "infobox"})

            if infobox:
                genre_row = infobox.find("th", string=re.compile("Genres?", re.I))
                if genre_row:
                    genre_cell = genre_row.find_next_sibling("td")
                    if genre_cell:
                        genres = [g.get_text(strip=True) for g in genre_cell.find_all(["a", "span"])]
                        # Clean up genres
                        cleaned = list(set([re.sub(r"\[\d+\]", "", genre.lower()) for genre in genres]))
                        
                        # Map genres to umbrella categories
                        mapped_genres = [merged_genres.get(genre, genre) for genre in cleaned]
                        
                        # Store in cache
                        wiki_genre_cache[artist_name] = list(set(mapped_genres))
                        return list(set(mapped_genres))
    except Exception as e:
        print(f"Error fetching genres for {artist_name}: {e}")

    wiki_genre_cache[artist_name] = []
    return []

# Apply to DataFrame
import pandas as pd

# Assuming 'artist_df' is the DataFrame with artist names
artist_df = pd.DataFrame({'name': ['Taylor Swift', 'Ariana Grande', 'Billie Eilish']})  # Example

artist_df['wiki_artist_genres'] = artist_df['name'].apply(get_genres_from_wikipedia)

# Save updated cache to a JSON file
cache_file = 'wiki_genre_cache.json'
with open(cache_file, "w") as f:
    json.dump(wiki_genre_cache, f)

# Save the final DataFrame to CSV
merged_csv = "wiki_artist_genres.csv"
artist_df.to_csv(merged_csv, index=False)
print(f"\nData with all merged genres saved to: {merged_csv}")


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


import json
import time
import requests
from spotipy import Spotify

# Load or initialize cache
CACHE_FILE = "wiki_genre_cache.json"
try:
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
except FileNotFoundError:
    cache = {}

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

"""
# -------------------------------
# Function to get artist metadata (with genres fallback)
def fetch_artist_metadata(artist_name, sp):
    try:
        if artist_name in cache:  # Check if the artist is already in the cache
            print(f"Cache hit for {artist_name}")
            artist_data = cache[artist_name]
            
            # Check if genres were missing and fetch if necessary
            if not artist_data.get('genres'):
                print(f"Genres missing for {artist_name}, fetching...")
                genres = fetch_genres(sp, artist_data["id"], artist_name)
                artist_data["genres"] = genres  # Update genres in cached data
                
                # Save the updated cache to file
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cache, f, indent=4)
        
            return artist_data
        
        result = sp.search(q=artist_name, type="artist", limit=5)
        best_match = None

        for artist in result['artists']['items']:
            if is_close_match(artist_name, artist['name']):
                best_match = artist
                break
        
        if best_match:
            artist_id = best_match['id']
            genres = fetch_genres(sp, artist_id, artist_name)  # Pass the artist_name here
            popularity = best_match['popularity']
            followers = best_match['followers']['total']
            
            metadata = {
                "id": artist_id,
                "name": best_match['name'],
                "genres": genres,
                "popularity": popularity,
                "followers": followers
            }
            
            # Save the metadata in the cache
            cache[artist_name] = metadata
            
            # Save the cache to a file
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=4)
            
            return metadata
        else:
            print(f"WARNING: No good match for {artist_name}")
            return None

    except Exception as e:
        print(f"Error fetching metadata for {artist_name}: {e}")
        return None
"""
# -------------------------------
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

################################
##### GETTING UPDATED ##########
##### GENRE INFO FROM WIKI #####
################################

merged_genres = {
    'pop': 'Pop',
    'bubblegum pop': 'Pop',
    'soft pop': 'Pop',
    'art pop': 'Pop',
    'classic rock': 'Rock',
    'rock': 'Rock',
    'emo': 'Rock',
    'emo pop': 'Rock',
    'pop punk': 'Rock',
    'alternative metal': 'Rock',
    'r&b': 'R&B',
    'neo soul': 'R&B',
    'soul': 'R&B',
    'hip hop': 'Hip Hop',
    'rap': 'Hip Hop',
    'hip-hop': 'Hip Hop',
    'dance': 'Electronic',
    'electronic': 'Electronic',
    'new age': 'Electronic',
    'house': 'Electronic',
    'techno': 'Electronic',
    'disco': 'Electronic',
    'hyperpop': 'Electronic',
    'country': 'Country',
    'latin pop': 'Latin',
    'latin': 'Latin',
    'folk': 'Folk',
    'celtic': 'Folk',
    'jazz': 'Standards/Traditional',
    'blues': 'Standards/Traditional',
    'adult standards': 'Standards/Traditional',
    'metal': 'Rock',
    'punk': 'Rock',
    'alternative': 'Rock',
    'indie': 'Pop',
    'funk': 'R&B',
    'dancehall': 'Hip Hop',
    'grunge': 'Rock',
    'new jack swing': 'R&B',
    'christmas': 'Seasonal',
    'soft rock': 'Rock',
    'variété française': 'International',
    'east coast hip hop': 'Hip Hop'
}

# Function to scrape genre from Wikipedia page
def get_genres_from_wikipedia(name):
    artist_name_url = name.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{artist_name_url}"

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve {name}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Try to find the infobox containing genre information
    infobox = soup.find("table", {"class": "infobox"})
    if infobox:
        genre_row = infobox.find("th", string="Genres")
        if genre_row:
            genres = genre_row.find_next("td").text.strip()
            return genres
    return None

# Function to clean, split, and map compound genre words
def split_and_map_genres(genre_list):
    # Define a pattern to match possible genres
    genre_patterns = [
        r'r&b', r'soul', r'pop', r'rock', r'hip hop', r'rap', r'jazz', r'dance', r'funk', r'country',
        r'blues', r'latin', r'folk', r'new wave', r'punk', r'alternative', r'electronic', r'disco',
        r'neo soul', r'countrypop', r'blue-eyed soul', r'post-grunge', r'art pop', r'house', r'funk-pop',
        r'synth-pop', r'pop rock', r'country pop', r'pop-punk', r'baroque pop', r'indie pop', r'r&b soul',
        r'emo', r'hip-hop', r'afrobeats', r'ska', r'k-pop', r'reggae', r'blues rock', r'pop rap'
    ]
    
    if isinstance(genre_list, str):
        # Match and replace the genres using the defined patterns
        for pattern in genre_patterns:
            genre_list = re.sub(pattern, f' {pattern} ', genre_list, flags=re.IGNORECASE)
        
        # Split the genres and remove any unnecessary whitespaces
        genres = re.split(r'\s+', genre_list.lower())
        genres = [g.strip() for g in genres if g]
        
        # Map genres to merged umbrella categories
        mapped_genres = [merged_genres.get(genre, genre) for genre in genres if genre in merged_genres]
        
        # Remove duplicates by converting to set and then back to list
        return list(set(mapped_genres))
    
    return []

# Scrape genres for each artist
artist_df['wiki_genres'] = artist_df['name'].apply(lambda artist: get_genres_from_wikipedia(artist))

# Clean, split, and map the genres immediately after scraping
artist_df['cleaned_genres'] = artist_df['wiki_genres'].apply(split_and_map_genres)

# Save the cleaned DataFrame to CSV
artist_df.to_csv("wiki_artist_genres_cleaned.csv", index=False)
print(f"Genres cleaned and saved to 'wiki_artist_genres_cleaned.csv'.")

"""
###################################
##### MERGE GENRES FROM ###########
##### BOTH WIKI AND SPOTIFY #######
###################################

# Define cleaning and combining function
def clean_and_combine_genres(row):
    # Combine Spotify genres and scraped genres
    genres = row['genres'] if isinstance(row['genres'], list) else []
    wiki_raw = row['wiki_genres'] if isinstance(row['wiki_genres'], str) else ""
    
    # Split wiki genres by common separators
    wiki_split = []
    for sep in ['\n', ',', ';']:
        if sep in wiki_raw:
            wiki_split = wiki_raw.split(sep)
            break
    if not wiki_split:
        # Fallback: lowercase and try to split by known patterns
        wiki_split = [wiki_raw]

    # Clean and normalize
    cleaned_wiki_genres = []
    for genre in wiki_split:
        genre = genre.strip().lower()
        genre = ''.join([c for c in genre if c.isalpha() or c == ' ' or c == '&' or c == '-'])  # Keep letters, spaces, '-', '&'
        if genre:
            cleaned_wiki_genres.append(genre)

    # Combine and deduplicate
    all_genres = set(genres + cleaned_wiki_genres)
    return list(all_genres)

# Create cleaned combined genre column
artist_df['artist_merged_genres'] = artist_df.apply(clean_and_combine_genres, axis=1)

# Drop the original 'genres' and 'wiki_genres' columns
artist_df = artist_df.drop(columns=['genres', 'wiki_genres'])

# Save to CSV
artist_df_file = "artist_merged_genres.csv"
current_directory = os.getcwd()
artist_df_file_path = os.path.join(current_directory, artist_df_file)
artist_df.to_csv(artist_df_file_path, index=False)

# Confirm
print(f"\nCleaned artist genre metadata saved to:\n'{artist_df_file_path}'")
print("\nUpdated columns:\n", artist_df.columns)

# TROUBLESHOOTING
# Show any artists with no genres even after wiki genre merge

# Ensure we treat empty lists and NaN properly
def is_empty_genre_list(val):
    return not isinstance(val, list) or len(val) == 0

# Filter the rows
no_genres_even_after_wiki_merge = artist_df[artist_df['wiki_artist_genres'].apply(is_empty_genre_list)].copy()

# Show results
print("\nArtists with no genre info even after Wiki merge:")
print(no_genres_even_after_wiki_merge[['name']])



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
df['Cluster'] = df_combined['Cluster']

# Filter DataFrame for each cluster using 'optimal_k' from elbow method
for cluster_num in range(optimal_k):
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



##############################
##### NETWORKX GRAPH FOR #####
##### ARTISTS IN CLUSTERS ####
##### EDGES ARE GENRES #######
##############################

import networkx as nx
import matplotlib.pyplot as plt

# Create a new graph
G = nx.Graph()

# Add nodes (artists) to the graph
for idx, row in df.iterrows():
    # Make sure genres is a list, and add each artist as a node with their genres and cluster
    G.add_node(row['name'], cluster=row['Cluster'], genres=row['genres'])

# Add edges based on genre overlap (for simplicity, we can just connect artists with at least one genre in common)
for i, artist1 in df.iterrows():
    for j, artist2 in df.iterrows():
        if i >= j:  # To avoid double-counting
            continue
        # Ensure genres are set types for intersection
        common_genres = set(artist1['genres']).intersection(set(artist2['genres']))
        if common_genres:
            # Add edge with the genres in common as the edge attribute
            G.add_edge(artist1['name'], artist2['name'], genres=list(common_genres))

# Visualize the graph with nodes colored by cluster
node_colors = [G.nodes[node]['cluster'] for node in G.nodes]

# Choose layout
layout = nx.spring_layout(G, seed=42)  # positions for all nodes

# Draw the graph
plt.figure(figsize=(15, 15))

# Draw nodes with colors based on clusters
nx.draw_networkx_nodes(G, layout, node_color=node_colors, cmap=plt.cm.tab10, node_size=500, alpha=0.7)

# Draw edges with gray color
nx.draw_networkx_edges(G, layout, alpha=0.7, width=1.0, edge_color='gray')

# Draw node labels
nx.draw_networkx_labels(G, layout, font_size=12, font_color='black')

# Add a title
plt.title("Artist Network by Clusters", fontsize=18)

# Display the plot
plt.axis('off')  # Turn off axis
plt.show()

"""
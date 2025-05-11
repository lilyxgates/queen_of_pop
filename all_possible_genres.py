import requests
import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import time  # For buffering, sleep function
from bs4 import BeautifulSoup
import pandas as pd
import yaml
import re

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


###########################################
##### EXTRACT POSSIBLE GENRES: Spotify ####
###########################################

# Load the CSV file with artist IDs
artist_df = pd.read_csv('spotify_artist_metadata.csv')


# Function to get genres from Spotify artist ID
def get_genres_from_artist_id(artist_id):
    try:
        artist_info = sp.artist(artist_id)
        return artist_info['genres']
    except Exception as e:
        print(f"Error fetching genres for artist {artist_id}: {e}")
        return []

# Get all genres for the selected artists
all_genres = []

# Loop through the artist IDs and get their genres
for artist_id in artist_df['id']:
    genres = get_genres_from_artist_id(artist_id)
    all_genres.extend(genres)

# Remove duplicates from the list of genres
spotify_genres = set(all_genres)

# Print out the unique genres
print(f"Unique Spotify genres from selected artists:\n")
print(spotify_genres)


#############################################
##### EXTRACT POSSIBLE GENRES: Wikipedia ####
#############################################


def scrape_wikipedia_genres(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    genre_list = []
    
    for item in soup.find_all(['li', 'td']):
        text = item.get_text().strip().lower()

        # Pattern can be updated/expanded as needed
        genre_pattern = r'\b(?:pop|rock|r&b|jazz|hip-hop|dance|electronic|country|soul|blues|metal|punk|alternative|indie|folk|classical|reggae|funk|dancehall|rap|grunge|latin|house|techno|disco|synth-pop)\b'
        
        matches = re.findall(genre_pattern, text)
        if matches:
            genre_list.extend(matches)

    return genre_list

# Wikipedia scraping
wiki_url = 'https://en.wikipedia.org/wiki/List_of_music_genres_and_styles'
all_wiki_genres = scrape_wikipedia_genres(wiki_url)
unique_wiki_genres = set(all_wiki_genres)

# Output
print(f"\nUnique Wikipedia genres scraped:\n")
print(unique_wiki_genres)

###############################################
##### MERGED GENRE MAPPING: Standardization ###
###############################################
# This mapping dictionary consolidates genres from Spotify and Wikipedia
# into broader umbrella categories (e.g., 'Pop', 'Rock', 'Hip Hop').
# It helps standardize artist classifications for analysis and clustering.

# Mapping Wikipedia + Spotify genres to umbrella categories
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

# Optional: print a list of all standardized genres
standardized_set = sorted(set(merged_genres.values()))
print(f"\nStandardized merged genre categories:\n{standardized_set}")

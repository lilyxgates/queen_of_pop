# Lily Gates
# Fri. 5/9/25

###############################
##### IMPORTING LIBRARIES  ####
###############################

import yaml  # Save keys
import os
import numpy as np 
import pandas as pd 

# For Spotify API
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials

import time  # for buffering, sleep function
from requests import get

# For KMeans Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# For data viz
import matplotlib.pyplot as plt

#####################################
##### IMPORT SPOTIFY KEY and URI ####
#####################################

# Display Current Directory and Read in API Key
# Assumes a file containing a dict of API key's is in the same dir as current open dir

# Current Directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# File with APIs
api_keys = "keys.yml"

# Construct full file path to API key dict
api_keys_path = os.path.join(current_dir, api_keys)

# Confirm Paths
print("-----------------------------------------------------------")
print(f"CURRENT DIRECTORY:\n{current_dir}")
print(f"API KEYS FILE PATH:\n{api_keys_path}")

# Read in File with Current API Key for Spotify's API
with open(api_keys_path, 'r') as file:
    keys = yaml.safe_load(file)

# Save true values for Client ID, Client Secret, and Spotify URI as own variables
spotify_id = keys['spotify_popqueen_key']  # Client ID
spotify_secret = keys['spotify_popqueen_pass']  # Client Secret
spotify_uri = keys['spotify_uri']

# Confirm key
print("-----------------------------------------------------------")
print(f"Client ID: {spotify_id}")
print(f"Client Secret: {spotify_secret}")
print(f"Spotify URI: {spotify_uri}")

#####################################
##### SPOTIFY API AUTHENTICATION ####
#####################################

sp_oauth = SpotifyOAuth(
    client_id=spotify_id,
    client_secret=spotify_secret,
    redirect_uri=spotify_uri,
    scope='user-library-read user-read-private'
)
token_info = sp_oauth.get_access_token(as_dict=False)
sp = spotipy.Spotify(auth=token_info)

#########################################
##### COLLECT SONG DATA FROM ARTISTS ####
#########################################

# Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=spotify_id,
                                               client_secret=spotify_secret,
                                               redirect_uri=spotify_uri,
                                               scope="user-library-read playlist-read-private"))

# Function to get audio features for a specific artist
def get_audio_features(artist_name):
    results = sp.search(q=artist_name, limit=20, type='track')
    tracks = results['tracks']['items']
    
    print(f"Found {len(tracks)} tracks for {artist_name}")

    features = []
    
    for track in tracks:
        track_id = track.get('id')
        track_name = track.get('name')
        print(f"Track: {track_name}, ID: {track_id}")
        
        if track_id is None:
            print("⚠️ Skipping: No ID")
            continue

        try:
            # Get audio features for the track
            audio_feature = sp.audio_features(track_id)[0]
            if audio_feature:
                print("✅ Feature collected")
                features.append(audio_feature)
            else:
                print("❌ Feature is None")
        except Exception as e:
            print(f"🚫 Error on track '{track_name}': {e}")
    
    return features

# Get audio features for Ariana Grande
ariana_features = get_audio_features("Ariana Grande")
print(f"Collected {len(ariana_features)} audio features")

# Print the features of the first track
if ariana_features:
    print(ariana_features[0])

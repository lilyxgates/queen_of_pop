import yaml
import os
import requests
import base64
import json
from urllib.parse import urlencode
from flask import Flask, request, redirect


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

# Replace with your actual Spotify Developer credentials
CLIENT_ID = spotify_id
CLIENT_SECRET = spotify_secret
REDIRECT_URI = 'http://127.0.0.1:9090/callback'
SCOPE = 'user-library-read playlist-read-private'

# Store the access and refresh tokens here (In production, save them securely)
access_token = None
refresh_token = None

# Flask app setup
app = Flask(__name__)

@app.route('/')
def login():
    auth_query = {
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'scope': SCOPE,
        'client_id': CLIENT_ID
    }

    url_args = urlencode(auth_query)
    auth_url = f"https://accounts.spotify.com/authorize?{url_args}"
    return redirect(auth_url)

@app.route('/callback')
def callback():
    global access_token, refresh_token
    code = request.args.get('code')

    token_url = 'https://accounts.spotify.com/api/token'
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    headers = {
        'Authorization': f'Basic {b64_auth_str}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    payload = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI
    }

    # Get the access token and refresh token
    response = requests.post(token_url, data=payload, headers=headers)
    token_data = response.json()

    access_token = token_data.get('access_token')
    refresh_token = token_data.get('refresh_token')

    # Return the access token and refresh token to the user
    return f"""
    <h1>Authorization successful!</h1>
    <p><strong>Access Token:</strong> {access_token}</p>
    <p><strong>Refresh Token:</strong> {refresh_token}</p>
    <p>Save these somewhere secure. You can now use the access token in your API requests.</p>
    """

def refresh_access_token():
    """Refresh the access token using the refresh token."""
    global access_token

    refresh_url = 'https://accounts.spotify.com/api/token'

    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token
    }

    headers = {
        'Authorization': f'Basic {base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.post(refresh_url, data=payload, headers=headers)
    token_data = response.json()

    # Update the access token
    access_token = token_data.get('access_token')
    print(f"New Access Token: {access_token}")
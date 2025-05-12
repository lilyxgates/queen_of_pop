# Billboard Top 100 Women Artists Clustering
*Written by Lily Gates*  
*May 2025*

## Description
This project clusters the Billboard Top 100 Women Artists of the 21st Century using Spotify metadata. It leverages Spotify’s Web API to collect artist information—such as genre affiliations, popularity scores, and follower counts—and applies KMeans clustering to uncover natural groupings among these artists. The ultimate goal is to analyze how these artists differ in audience reception, genre affiliation, and other musical characteristics.

## Usage
- **Spotify API Integration**: Authenticates via OAuth and collects metadata using the spotipy Python client.
- **Artist Metadata Collection**: Pulls artist IDs, popularity scores, follower counts, and genre tags.
- **Data Cleaning and Feature Engineering**:
    - One-hot encoding of genres
    - Normalization of numerical features (ranking, popularity, followers)
- **KMeans Clustering**: Uses the elbow method to determine the optimal number of clusters and groups artists accordingly.
- **Cluster Analysis**: Prints artist lists by cluster and summarizes top genres per cluster.

## Methodology

### 1. Data Collection
Artists are pulled from Billboard's ranked list.  
* For each artist, metadata is pulled using `sp.search()` and `sp.artist()` endpoints from the Spotify API.

### 2. Preprocessing
* Genres are one-hot encoded for machine readability.
* Numerical features are standardized (z-score normalization).
* Data is concatenated into a single dataframe for clustering.

### 3. Clustering
* KMeans is applied with varying K (1–30).
* The elbow method is used to determine the best K visually.
* Final KMeans model is fit and each artist is assigned a cluster.

### 4. Analysis
* Artists grouped by cluster are printed for interpretation.
* Genre frequencies per cluster are computed to identify dominant themes.

## Required Dependencies
- `spotipy`: Spotify Web API wrapper
- `pandas`, `numpy`: Data manipulation
- `sklearn`: KMeans clustering
- `matplotlib`, `seaborn`: Visualization
- `yaml`: Credentials handling

## Output
The script outputs:
- A DataFrame (`df_combined`) containing cleaned features and cluster labels.
- Printed lists of artists per cluster (with rankings).
- Elbow method plot showing inertia vs. K.
- Cluster-level genre frequency summaries (top genres per cluster).

## Limitations
- Genre tagging is based on Spotify's metadata, which may be inconsistent or overly broad.
- API rate limits may introduce pauses or delays.
- Only artists with retrievable Spotify IDs are included.

## Future Improvements
- Incorporate audio features from top tracks for deeper analysis.
- Perform PCA or t-SNE for cluster visualization in 2D/3D.
- Add gender comparative analysis (e.g., comparing top female and male artists).
- Export results as interactive dashboards (e.g., using Plotly or Streamlit).

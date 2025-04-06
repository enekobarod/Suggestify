import sqlite3
import pandas as pd
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt
import psutil

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import requests
from io import BytesIO
from PIL import Image


class RecommenderModel:
    def __init__(self, min_ratings_needed=20):
        self.min_ratings_needed = min_ratings_needed
        
        ####################LOAD
        venv_path = os.path.dirname(os.path.dirname(sys.executable))
        project_path = os.path.dirname(venv_path)
        db_path = os.path.join(project_path, "extracted.db")

        sqlite_conn = sqlite3.connect(db_path)
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in SQLite:", tables)

        df_og = pd.read_sql("SELECT * FROM extracted", sqlite_conn)
        sqlite_conn.close()


        ####################TRANSFORM
        df_og['track_id'] = df_og['track_uri'].astype('category').cat.codes
        df_og['artist_id'] = df_og['artist_name'].astype('category').cat.codes
        df_og['album_id'] = df_og['album_name'].astype('category').cat.codes

        #main df with relevant columns
        self.df = df_og[[
            "track_id", "track_name", "artist_name", "artist_id", "album_name", "album_id",
            "duration_ms", "danceability", "energy", "key", "loudness", "mode", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"
        ]].copy()

        #numerical features
        self.df_numerical = self.df[[
            "track_id", "artist_id", "album_id", "duration_ms", "danceability", "energy", "key",
            "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "time_signature"
        ]].copy()

        #numerical features without track_id
        numerical_features = [
            "artist_id", "album_id", "duration_ms", "danceability", "energy", "key", "loudness",
            "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
            "tempo", "time_signature"
        ]

        ####################Text features
        all_titles = self.df['track_name'].astype(str).tolist()
        self.title_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=20 #this selects the top 20 words (con más peta) and sees how relevant those 20 words are for the title
        )
        title_vectors = self.title_vectorizer.fit_transform(all_titles)
        #converting to a dense array for combining with numeric
        title_vectors_dense = title_vectors.toarray()  # (num_tracks, up to 200)

        #convining numeric and text features
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(self.df_numerical[numerical_features])
        #stack horizontally
        combined_features = np.hstack((numeric_scaled, title_vectors_dense))

        pca = PCA(n_components=8)
        self.reduced_features_full = pca.fit_transform(combined_features).astype(np.float32)

        self.nn_model_full = NearestNeighbors(
            n_neighbors=50,
            metric='cosine',
            algorithm='auto',
            n_jobs=-1
        )
        self.nn_model_full.fit(self.reduced_features_full)

        self.ratings = pd.DataFrame(columns=["track_id", "rating"])

        self.gesture_to_rating = {
            "right": 1,
            "left": -1,
            "up": 0,
            "double_click": 2
        }

        #api spotify
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id="1e6d532d63fa4fbb82230cd1ffd06fc9",
            client_secret="b6281e5bb26c43d282e1a4987828366c"
        ))

        print("Model initialized. Minimum ratings required:", self.min_ratings_needed)


    def get_random_song(self):
        idx = random.randint(0, len(self.df) - 1)
        row = self.df.iloc[idx]
        image = self.get_cover_image(row["track_name"], row["artist_name"])

        return {
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "artist_name": row["artist_name"],
            "cover_image": image
        }

    def get_cover_image(self, track_name, artist_name):
        try:
            query = f"track:{track_name} artist:{artist_name}"
            results = self.sp.search(q=query, type='track', limit=1)
            items = results.get("tracks", {}).get("items", [])
            if items:
                image_url = items[0]["album"]["images"][0]["url"]  #thisis the one with the highest resolution
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                return img
        except Exception as e:
            print(f"Error with the cover of '{track_name}' - '{artist_name}': {e}")

        return Image.open("caratula.jpg")

    def submit_gesture_rating(self, track_id, gesture):
        if gesture not in self.gesture_to_rating:
            print(f"Unknown gesture: {gesture}")
            return

        rating_value = self.gesture_to_rating[gesture]
        new_row = {"track_id": track_id, "rating": rating_value}
        self.ratings = pd.concat([self.ratings, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Rating stored: track_id={track_id}, gesture={gesture}, numeric={rating_value}")

    def has_enough_ratings(self):
        return len(self.ratings) >= self.min_ratings_needed


    def get_final_recommendations(self, top_n=5, diversity=0.3):
        if len(self.ratings) == 0:
            return pd.DataFrame(columns=self.df.columns)

        df_reset = self.df.reset_index(drop=True)
        track_id_list = df_reset['track_id'].tolist()
        track_id_to_idx = {tid: i for i, tid in enumerate(track_id_list)}

        #score array for each track
        all_scores = np.zeros(len(df_reset))

        #for each rated track, find neighbors and accumulate similarity * rating
        for _, row in self.ratings.iterrows():
            rated_track_id = row["track_id"]
            rating_val = row["rating"]

            if rated_track_id not in track_id_to_idx:
                continue

            tidx = track_id_to_idx[rated_track_id]
            distances, indices = self.nn_model_full.kneighbors([self.reduced_features_full[tidx]])
            #convert distances to similarity
            similarities = 1 / (1 + distances[0])  # shape: (50,)

            #weighted by the rating
            all_scores[indices[0]] += similarities * rating_val

        #we add some random noise (for diversity)
        noise = (np.random.rand(len(all_scores)) - 0.5) * 2 * diversity * np.mean(np.abs(all_scores))
        final_scores = all_scores * (1 - diversity) + noise

        #and we exclude tracks already rated
        for _, row in self.ratings.iterrows():
            exclude_id = row["track_id"]
            if exclude_id in track_id_to_idx:
                final_scores[track_id_to_idx[exclude_id]] = -np.inf

        #sort in descending order
        top_indices = np.argsort(final_scores)[-top_n:][::-1]
        valid_indices = top_indices[final_scores[top_indices] > -np.inf]

        if len(valid_indices) == 0:
            return pd.DataFrame(columns=self.df.columns)  #no recommendations

        return df_reset.iloc[valid_indices].copy()

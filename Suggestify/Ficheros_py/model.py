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


class RecommenderModel:
    """
    Class that:
      - Loads the entire dataset from 'extracted.db'
      - Prepares it (df, numeric features, scaling, PCA)
      - Builds a NearestNeighbors model
      - Allows random track selection
      - Stores user ratings
      - Returns final recommendations once enough songs are rated
    """

    def __init__(self, min_ratings_needed=20):
        """
        - Connect to DB and load data
        - Transform columns into IDs
        - Scale features, do PCA
        - Build NearestNeighbors model
        - Prepare a DataFrame for storing user ratings
        :param min_ratings_needed: how many ratings we want before generating final recommendations
        """
        self.min_ratings_needed = min_ratings_needed

        #################### LOAD FROM DB
        # Attempt to locate your DB. Adjust as needed.
        venv_path = os.path.dirname(os.path.dirname(sys.executable))
        project_path = os.path.dirname(venv_path)
        db_path = os.path.join(project_path, "extracted.db")

        # If needed, switch between your known paths:
        #db_path = venv_path + "/extracted.db"
        #db_path = project_path + "/extracted.db"

        sqlite_conn = sqlite3.connect(db_path)
        
        
        
        #####################################################
        venv_path = os.path.dirname(os.path.dirname(sys.executable))
        #conseguir direccion del proyecto, donde esta la bd
        project_path = os.path.dirname(venv_path)

        #sqlite_conn = sqlite3.connect(project_path+"/extracted.db") #clase
        sqlite_conn = sqlite3.connect(venv_path+"/extracted.db") #portatil
        #####################################################
        
        
        
        
        
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in SQLite:", tables)

        df_og = pd.read_sql("SELECT * FROM extracted", sqlite_conn)
        sqlite_conn.close()

        # Just for demonstration:
        print("Loaded dataset shape:", df_og.shape)

        #################### TRANSFORM
        df_og['track_id'] = df_og['track_uri'].astype('category').cat.codes
        df_og['artist_id'] = df_og['artist_name'].astype('category').cat.codes
        df_og['album_id'] = df_og['album_name'].astype('category').cat.codes

        # We keep a main df with relevant columns
        self.df = df_og[[
            "track_id", "track_name", "artist_name", "artist_id", "album_name", "album_id",
            "duration_ms", "danceability", "energy", "key", "loudness", "mode", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"
        ]].copy()

        # We store also "df_numerical"
        self.df_numerical = self.df[[
            "track_id", "artist_id", "album_id", "duration_ms", "danceability", "energy", "key",
            "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "time_signature"
        ]].copy()

        # Full set of numerical features (excluding track_id)
        numerical_features = [
            "artist_id", "album_id", "duration_ms", "danceability", "energy", "key", "loudness",
            "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
            "tempo", "time_signature"
        ]

        #################### SCALING
        scaler = StandardScaler()
        df_scaled_full = scaler.fit_transform(self.df_numerical[numerical_features])

        #################### PCA
        pca = PCA(n_components=4)
        self.reduced_features_full = pca.fit_transform(df_scaled_full).astype(np.float32)

        #################### NEAREST NEIGHBORS MODEL
        self.nn_model_full = NearestNeighbors(
            n_neighbors=50,
            metric='cosine',
            algorithm='auto',
            n_jobs=-1
        ).fit(self.reduced_features_full)

        #################### RATINGS
        # DataFrame with columns: track_id, rating
        # rating = -1 (dislike), 0 (skip), 1 (like), 2 (super like) if you want
        self.ratings = pd.DataFrame(columns=["track_id", "rating"])

        # For gestures, we’ll map “left” => -1, “right” => +1, “up” => 0, etc.
        self.gesture_to_rating = {
            "right": 1,
            "left": -1,
            "up": 0,
            # you can add "super like" = 2 if you have a special gesture
        }

        print("RecommenderModel initialized. Ready to rate up to", self.min_ratings_needed)

    def get_random_song(self):
        """
        Returns a dict with info for a random song from self.df,
        including which 'track_id' it has, the 'track_name', the 'artist_name',
        and a placeholder 'image_path' to display in the GUI.
        """
        idx = random.randint(0, len(self.df) - 1)
        row = self.df.iloc[idx]

        return {
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "artist_name": row["artist_name"],
            # If you have real cover art, replace the string below with the real file path
            "image_path": "caratula.jpg"
        }

    def submit_gesture_rating(self, track_id, gesture):
        """
        Convert the user gesture (left/right/up) into a numeric rating,
        store it in 'self.ratings'.
        """
        if gesture not in self.gesture_to_rating:
            # Unknown gesture, ignore or handle error
            print(f"Warning: gesture '{gesture}' is not in gesture_to_rating map.")
            return

        rating_value = self.gesture_to_rating[gesture]

        # Add a new row to self.ratings
        new_row = {"track_id": track_id, "rating": rating_value}
        self.ratings = pd.concat([self.ratings, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Stored rating: track_id={track_id}, gesture={gesture}, numeric={rating_value}")

    def has_enough_ratings(self):
        """
        Check if user has rated at least 'min_ratings_needed' times.
        """
        return len(self.ratings) >= self.min_ratings_needed

    def get_final_recommendations(self, top_n=5, diversity=0.3):
        """
        Once we have enough ratings, run the content-based recommendation.
        Return a DataFrame with the recommended tracks (track_name, artist_name, etc.)

        This replicates the same approach as your original function but
        adapted to use 'self.ratings', 'self.df', 'self.nn_model_full', etc.
        """
        # If no ratings, just return an empty frame
        if len(self.ratings) == 0:
            return pd.DataFrame(columns=self.df.columns)

        # Re-build track_id -> index mapping
        df_reset = self.df.reset_index(drop=True)
        track_id_list = df_reset['track_id'].tolist()
        track_id_to_idx = {tid: i for i, tid in enumerate(track_id_list)}

        # We'll accumulate a "score" for each track
        all_scores = np.zeros(len(df_reset))

        # For each user rating, find nearest neighbors
        for _, row in self.ratings.iterrows():
            track_id = row["track_id"]
            rating_val = row["rating"]

            if track_id not in track_id_to_idx:
                continue

            tidx = track_id_to_idx[track_id]

            distances, indices = self.nn_model_full.kneighbors(
                [self.reduced_features_full[tidx]]
            )  # shape: (1, 50) for both
            # convert distances -> similarities
            similarities = 1 / (1 + distances[0])  # shape (50,)

            # Weighted by rating
            all_scores[indices[0]] += similarities * rating_val

        # Diversity: add some random noise
        noise = (np.random.rand(len(all_scores)) - 0.5) * 2 * diversity * np.mean(np.abs(all_scores))
        final_scores = all_scores * (1 - diversity) + noise

        # Exclude tracks already rated
        for _, row in self.ratings.iterrows():
            exclude_id = row["track_id"]
            if exclude_id in track_id_to_idx:
                final_scores[track_id_to_idx[exclude_id]] = -np.inf

        # Sort descending
        top_indices = np.argsort(final_scores)[-top_n:][::-1]
        valid_indices = top_indices[final_scores[top_indices] > -np.inf]

        if len(valid_indices) == 0:
            return pd.DataFrame(columns=self.df.columns)  # no recommendations

        return df_reset.iloc[valid_indices].copy()


# If you want to test model.py alone (without GUI), you can do something like:
#  (But typically you only import RecommenderModel in GUI.)
if __name__ == "__main__":
    model = RecommenderModel(min_ratings_needed=5)

    # Example test:
    print("Random track:", model.get_random_song())
    # Rate it:
    test_track = model.get_random_song()
    model.submit_gesture_rating(test_track["track_id"], "right")

    # Check if enough ratings:
    print("Enough ratings so far?", model.has_enough_ratings())
    # You'd do more tests here...

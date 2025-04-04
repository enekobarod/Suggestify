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
 
    def __init__(self, min_ratings_needed=20):
        #load data
        #Transform columns
        #Scale features, PCA
        #NearestNeighbors
        #df for storing user ratings
        #set how many ratings we want before generating the final recommendations

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

        print("Shape:", df_og.shape)

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

        ####################SCALE
        scaler = StandardScaler()
        df_scaled_full = scaler.fit_transform(self.df_numerical[numerical_features])

        ####################PCA
        pca = PCA(n_components=4)
        self.reduced_features_full = pca.fit_transform(df_scaled_full).astype(np.float32)

        ####################NEAREST NEIGHBORS MODEL
        self.nn_model_full = NearestNeighbors(
            n_neighbors=50,
            metric='cosine',
            algorithm='auto',
            n_jobs=-1
        ).fit(self.reduced_features_full)

        ####################RATINGS
        #df with columns track_id, rating
        self.ratings = pd.DataFrame(columns=["track_id", "rating"])

        self.gesture_to_rating = {
            "right": 1,
            "left": -1,
            "up": 0,
            #ainadir si aquí si ponemos superlike
        }

        print("Inicializado. Mínimo de ratings:", self.min_ratings_needed)

    def get_random_song(self):
        #Devuelve in diccionario con info de una canción raandom de self.df
        idx = random.randint(0, len(self.df) - 1)
        row = self.df.iloc[idx]

        return {
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "artist_name": row["artist_name"],
            #TENEMOS QUE CAMBIAR ESTO Y PONER LA CARATULA REAL
            "image_path": "caratula.jpg"
        }

    def submit_gesture_rating(self, track_id, gesture):
        #convertir gesto "left", "right"... en número y guardarlo
        if gesture not in self.gesture_to_rating:
            #unknown gesture
            print(f"error")
            return

        rating_value = self.gesture_to_rating[gesture]

        new_row = {"track_id": track_id, "rating": rating_value}
        self.ratings = pd.concat([self.ratings, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Rating stored: track_id={track_id}, gesture={gesture}, numeric={rating_value}")

    def has_enough_ratings(self):
        #ver si el usuario ha valorado la cantidad necesaria de canciones (min_ratings_needed)
        return len(self.ratings) >= self.min_ratings_needed

    """
    def get_final_recommendations(self, top_n=5, diversity=0.3):
        #Una vez tengamos la cantidad necesaria de ratings, corremos el modelo y devolvemos
            #un dataframe con las canciones recomendadas (obtenidas desde 'self.ratings', 'self.df'...)
            
        if len(self.ratings) == 0:
            return pd.DataFrame(columns=self.df.columns) #vacio

        #get track_id
        df_reset = self.df.reset_index(drop=True)
        track_id_list = df_reset['track_id'].tolist()
        track_id_to_idx = {tid: i for i, tid in enumerate(track_id_list)}

        #store a score for each song
        all_scores = np.zeros(len(df_reset))

        #for each rating, find the nearest neighbors
        for _, row in self.ratings.iterrows():
            track_id = row["track_id"]
            rating_val = row["rating"]

            if track_id not in track_id_to_idx:
                continue

            tidx = track_id_to_idx[track_id]

            distances, indices = self.nn_model_full.kneighbors(
                [self.reduced_features_full[tidx]]
            )  #(1, 50)
            #convert distances into similarities
            similarities = 1 / (1 + distances[0])  #(50,)

            #weighted by rating
            all_scores[indices[0]] += similarities * rating_val

        #Add some random noise (for diversity)
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

    """

    #Modelo mejorado para prueba
    def get_final_recommendations(self, top_n=5, diversity=0.3):
       
        if len(self.ratings) == 0:
            return pd.DataFrame(columns=self.df.columns)

        df_reset = self.df.reset_index(drop=True)
        n_tracks = len(df_reset)
        
        weighted_scores = np.zeros(n_tracks, dtype=np.float64)
        similarity_sums = np.zeros(n_tracks, dtype=np.float64)
        
        track_id_list = df_reset['track_id'].tolist()
        track_id_to_idx = {tid: i for i, tid in enumerate(track_id_list)}

        for _, row in self.ratings.iterrows():
            track_id = row["track_id"]
            rating_val = row["rating"]

            if track_id not in track_id_to_idx:
                continue

            tidx = track_id_to_idx[track_id]


            distances, indices = self.nn_model_full.kneighbors(
                [self.reduced_features_full[tidx]]
            )
            #distances, indices each have shape (1, n_neighbors)
            distances = distances[0]  # shape (n_neighbors,)
            indices = indices[0]      # shape (n_neighbors,)

            similarities = 1.0 / (1.0 + distances)  # shape (n_neighbors,)

            weighted_scores[indices] += similarities * rating_val
            similarity_sums[indices] += similarities

        epsilon = 1e-9
        final_scores = np.where(
            similarity_sums > 0,
            weighted_scores / (similarity_sums + epsilon),
            -np.inf
        )

        valid_mask = final_scores != -np.inf
        if np.any(valid_mask):
            mean_abs_score = np.mean(np.abs(final_scores[valid_mask]))
        else:
            mean_abs_score = 0.0

        noise = (np.random.rand(n_tracks) - 0.5) * 2.0 * diversity * mean_abs_score
        final_scores[valid_mask] = (
            final_scores[valid_mask] * (1.0 - diversity) + noise[valid_mask]
        )

        for _, row in self.ratings.iterrows():
            exclude_id = row["track_id"]
            if exclude_id in track_id_to_idx:
                final_scores[track_id_to_idx[exclude_id]] = -np.inf

        top_indices = np.argsort(final_scores)[::-1]

        top_indices = top_indices[final_scores[top_indices] != -np.inf]

        top_indices = top_indices[:top_n]

        if len(top_indices) == 0:
            return pd.DataFrame(columns=self.df.columns)

        return df_reset.iloc[top_indices].copy()

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
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from io import BytesIO
from PIL import Image
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


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
            "double_click": 2
        }

        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id="1e6d532d63fa4fbb82230cd1ffd06fc9",
            client_secret="b6281e5bb26c43d282e1a4987828366c"
        ))

        print("Inicializado. Mínimo de ratings:", self.min_ratings_needed)

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
        plot_folder = os.path.join(os.getcwd(), "images")
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
        return Image.open(str(plot_folder+'/caratula.jpg'))

    def submit_gesture_rating(self, user_id, track_id, gesture):
        #convertir gesto "left", "right"... en número y guardarlo
        if gesture not in self.gesture_to_rating:
            #unknown gesture
            print(f"error")
            return

        rating_value = self.gesture_to_rating[gesture]

        new_row = {"track_id": track_id, "rating": rating_value}
        self.ratings = pd.concat([self.ratings, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Rating stored: track_id={track_id}, gesture={gesture}, numeric={rating_value}")


        with open("register.csv", "a") as f:
            f.write(f"{user_id},{track_id},{rating_value}\n")

    def has_enough_ratings(self):
        #ver si el usuario ha valorado la cantidad necesaria de canciones (min_ratings_needed)
        n_valorada= 0
        for _, row in self.ratings.iterrows():
            rating_val = row["rating"]
            if(rating_val!=0):
                n_valorada+=1
        return n_valorada >= self.min_ratings_needed
    
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

            distances, indices = self.nn_model_full.kneighbors([self.reduced_features_full[tidx]])  #(1, 50)
            #convert distances into similarities
            similarities = 1 / (1 + distances[0])  #(50,)

            #weighted by rating
            all_scores[indices[0]] += similarities * rating_val

        #We add some random noise (for diversity)
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
    
    def get_user_2D_position(self):
        if len(self.ratings) == 0:
            #No ratings -> no user location
            return None

        #build a map from track_id to row index
        track_id_to_idx = {
            self.df_numerical.iloc[i]['track_id']: i
            for i in range(len(self.df_numerical))
        }

        user_pref_vector = np.zeros(self.reduced_features_full.shape[1], dtype=np.float32)
        total_weight = 0.0

        for _, row in self.ratings.iterrows():
            track_id = row["track_id"]
            rating_val = row["rating"]
            if track_id not in track_id_to_idx:
                continue
            tidx = track_id_to_idx[track_id]
            user_pref_vector += self.reduced_features_full[tidx] * rating_val
            total_weight += abs(rating_val)

        if total_weight == 0:
            return None

        user_pref_vector /= total_weight
        return user_pref_vector[:2]

    def save_user_plot(self, filename="user_plot.png"):
        plt.figure()
        plt.scatter(
            self.reduced_features_full[:, 0],
            self.reduced_features_full[:, 1],
            alpha=0.5,
            color="gray"
        )

        #plot user location
        user_pos = self.get_user_2D_position()
        if user_pos is not None:
            plt.scatter(user_pos[0], user_pos[1], color="#006400", s=100)

        plt.title("First two PCA components")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"User plot saved to {filename}")   

    def save_user_like_evolution_plot(self, user_id, filename="like_evolution.png"):

        if not os.path.exists("register.csv"):
            print("No register.csv found")
            return

        df_reg = pd.read_csv("register.csv", header=None, names=["user_id", "track_id", "rating"])
        df_user = df_reg[df_reg["user_id"] == user_id].reset_index(drop=True)
        if df_user.empty:
            print(f"No ratings for user {user_id}")
            return

        # For each row, define a new column: is_positive (1 if rating=1 or 2)
        # is_negative (1 if rating=-1), ignoring 0
        df_user["is_positive"] = df_user["rating"].apply(lambda r: 1 if r in [1,2] else 0)
        df_user["is_negative"] = df_user["rating"].apply(lambda r: 1 if r == -1 else 0)

        df_user["pos_cum"] = df_user["is_positive"].cumsum()
        df_user["neg_cum"] = df_user["is_negative"].cumsum()

        plt.figure()
        plt.plot(df_user.index, df_user["pos_cum"], label="Cumulative Likes/Superlikes", color="green")
        plt.plot(df_user.index, df_user["neg_cum"], label="Cumulative Dislikes", color="red")
        plt.title(f"Like/Dislike Evolution")
        plt.xlabel("Rating # (chronological)")
        plt.ylabel("Cumulative Count")
        plt.legend()
        plt.grid(True)

        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Like evolution plot saved to {filename}")

    def save_user_gesture_distribution_plot(self, user_id, filename="gesture_distribution.png"):
        if not os.path.exists("register.csv"):
            print("No register.csv found")
            return

        df_reg = pd.read_csv("register.csv", header=None, names=["user_id", "track_id", "rating"])
        df_user = df_reg[df_reg["user_id"] == user_id].reset_index(drop=True)
        if df_user.empty:
            print(f"No ratings for user {user_id}")
            return

        #count how many times each rating was given
        gesture_counts = df_user["rating"].value_counts().sort_index()

        rating_map = {
            -1: "Dislike (-1)",
             0: "Skip (0)",
             1: "Like (1)",
             2: "Superlike (2)"
        }

        x_labels = [rating_map.get(r, str(r)) for r in gesture_counts.index]

        plt.figure()
        plt.bar(x_labels, gesture_counts.values)
        plt.title(f"Gesture Distribution for User {user_id}")
        plt.xlabel("Gesture Type")
        plt.ylabel("Count")

        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Gesture distribution plot saved to {filename}")
    
    def save_user_ratings_plot(self, user_id, filename="users_plot.png"):
        id_list = {row["track_id"]: row["rating"] for _, row in self.ratings.iterrows()}
        ids = list(id_list.keys())
        ratings = list(id_list.values())

        filtered_features = self.reduced_features_full[ids]

        color_map = {-1: 'darkred', 0: 'gray', 1: 'yellow', 2: 'green'}
        colors = [color_map[rating] for rating in ratings]

        plt.figure()

        plt.scatter(
            filtered_features[:, 0],  
            filtered_features[:, 1],  
            c=colors,  
            alpha=0.7
        )

        user_pos = self.get_user_2D_position()
        if user_pos is not None:
            plt.scatter(user_pos[0], user_pos[1], color="#006400", s=100)

        full_x = self.reduced_features_full[:, 0]
        full_y = self.reduced_features_full[:, 1]
        plt.xlim(full_x.min() - 1, full_x.max() + 1)
        plt.ylim(full_y.min() - 1, full_y.max() + 1)

        plt.title("First two PCA components")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"User ratings plot saved to {filename}")

    
    
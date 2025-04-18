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
from datetime import date


class RecommenderModel:
 
    def __init__(self, min_ratings_needed=20):
        self.min_ratings_needed = min_ratings_needed

        """
        # Determine the project path
        venv_path = os.path.dirname(os.path.dirname(sys.executable))
        project_path = os.path.dirname(venv_path)
        db_path = os.path.join(project_path, "extracted.db")

        # Alternate path setup (likely for local script execution)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        db_path = os.path.join(project_path, "extracted.db")
        """
        ####################LOAD
        venv_path = os.path.dirname(os.path.dirname(sys.executable))
        project_path = os.path.dirname(venv_path)
        
        
        #nora
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        #nora
        
        db_path = os.path.join(project_path, "extracted.db")

        # Connect to the SQLite database
        sqlite_conn = sqlite3.connect(db_path)
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in SQLite:", tables)

        # Load the main dataset
        df_og = pd.read_sql("SELECT * FROM extracted", sqlite_conn)
        sqlite_conn.close()

        print("Shape:", df_og.shape)

        ####################TRANSFORM
        # Encoding categorical features as numeric codes
        df_og['track_id'] = df_og['track_uri'].astype('category').cat.codes
        df_og['artist_id'] = df_og['artist_name'].astype('category').cat.codes
        df_og['album_id'] = df_og['album_name'].astype('category').cat.codes

        # Define numerical features
        numerical_features = [
            "artist_id", "album_id", "duration_ms", "danceability", "energy", "key", "loudness",
            "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
            "tempo", "time_signature"
        ]
        
        # Include ratings dataset
        ratings_file = os.path.join(project_path, 'top_albums.csv')
        df_ratings = pd.read_csv(ratings_file, delimiter=",", na_values="NaN")

        # Rename 'artist' column to 'artist_name'
        df_ratings = df_ratings.rename(columns={'artist': 'artist_name'})

        # Keep only necessary columns
        df_ratings = df_ratings[['album_name', 'artist_name', 'releasedate', 'total_rating', 'total_reviews', 'pr_genres', 'sec_genres', 'tags']]

        # Select relevant columns for merging
        selected_columns = [
            "track_id", "artist_id", "track_name", "artist_name", "album_name", "album_id", 
            "duration_ms", "danceability", "energy", "key", "loudness", "mode", "speechiness", 
            "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"
        ]
        
        # Create the main dataframe with selected columns
        self.df = df_og.copy()

        # Merge the ratings dataset with the original dataset
        df_ratings = df_og.merge(
            df_ratings,
            on=['album_name', 'artist_name'],
            how='right'
        )
        
        # Combine primary and secondary genres
        df_ratings['genres'] = (
            df_ratings['pr_genres'].fillna('') + ', ' + df_ratings['sec_genres'].fillna('')
        ).str.strip(', ').str.replace(r',\s+,', ', ', regex=True)

        # Drop the primary and secondary genre columns after merging
        df_ratings.drop(columns=['pr_genres', 'sec_genres'], inplace=True)

        # Clean up and sort the 'genres' column
        df_ratings['genres'] = (
            df_ratings['genres']
            .str.split(', ')
            .apply(lambda x: ', '.join(sorted(set(x))) if x != [''] else np.nan)
        )

        # Drop rows with missing track_id or track_name
        df_ratings = df_ratings.dropna(subset=['track_id'])
        df_ratings = df_ratings.dropna(subset=['track_name'])

        # Sort by total reviews in descending order
        df_ratings.sort_values(by='total_reviews', ascending=False, inplace=True)

        # Parse the release date into a numeric format (YYYYMM)
        def parse_release_date(date_str):
            try:
                date = pd.to_datetime(date_str, errors='raise')
                if pd.notna(date):
                    return date.strftime('%Y%m')
                else:
                    return None
            except Exception as e:
                print(f"Error parsing date '{date_str}': {e}")
                return None

        # Apply the date parsing function
        df_ratings['releasedate'] = df_ratings['releasedate'].apply(parse_release_date)

        # Update the numerical features list to include additional features
        numerical_features += ["releasedate", "total_rating", "total_reviews"]

        # Create dummy variables for genres and tags
        genre_dummies = df_ratings['genres'].str.get_dummies(sep=', ')
        genre_columns = genre_dummies.columns.tolist()
        
        tags_dummies = df_ratings['tags'].str.get_dummies(sep=', ')
        tags_columns = tags_dummies.columns.tolist()

        df_ratings = pd.concat([df_ratings, genre_dummies, tags_dummies], axis=1)

        # Final list of numerical features including genre and tag dummies
        numerical_features += genre_columns
        numerical_features += tags_columns

        

        # Create the numerical features dataframe
        columns = ["track_id", "track_name", "artist_name", "album_name"] + numerical_features
        self.df_numerical = df_ratings[columns].sort_values(by='total_reviews', ascending=False).copy()

        print(self.df_numerical.head())
        
        
        
        
        
        


        ####################SCALE
        scaler = StandardScaler()
        df_scaled_full = scaler.fit_transform(self.df_numerical[numerical_features])

        ####################PCA
        pca = PCA(n_components=4)
        self.reduced_features_full = pca.fit_transform(df_scaled_full).astype(np.float32)

        print(pca.explained_variance_ratio_)

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
        # Inicialización solo la primera vez
        if not hasattr(self, 'populares_albums'):
            # Obtenemos el índice de la canción más popular por álbum
            idx = self.df_numerical.groupby('album_id')['total_reviews'].idxmax()
            # Creamos el dataframe filtrado y ordenado
            self.populares_albums = self.df_numerical.loc[idx].sort_values(
                'total_reviews', 
                ascending=False
            ).reset_index(drop=True)
            self.current_index = 0

        # Obtenemos la canción actual
        row = self.populares_albums.iloc[self.current_index % len(self.populares_albums)]
        self.current_index += 1
        
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
        return len(self.ratings) >= self.min_ratings_needed
    
    def get_final_recommendations(self, top_n=5, diversity=0.3):

        if len(self.ratings) == 0:
            return pd.DataFrame(columns=self.df.columns)  # vacío

        # Usamos df_ratings porque fue el dataset usado para entrenar PCA y NearestNeighbors
        df_reset = self.df_numerical.reset_index(drop=True)
        track_id_list = self.df_numerical.index.tolist()  # El índice ya es numérico y coincide con self.reduced_features_full
        track_id_to_idx = {self.df_numerical.iloc[i]['track_id']: i for i in range(len(self.df_numerical))}

        # Inicializamos el vector de puntuaciones
        all_scores = np.zeros(len(df_reset))

        # Recorremos cada rating
        for _, row in self.ratings.iterrows():
            track_id = row["track_id"]
            rating_val = row["rating"]

            if track_id not in track_id_to_idx:
                continue

            tidx = track_id_to_idx[track_id]

            distances, indices = self.nn_model_full.kneighbors([self.reduced_features_full[tidx]])
            similarities = 1 / (1 + distances[0])
            all_scores[indices[0]] += similarities * rating_val

        # Añadimos ruido para diversidad
        noise = (np.random.rand(len(all_scores)) - 0.5) * 2 * diversity * np.mean(np.abs(all_scores))
        final_scores = all_scores * (1 - diversity) + noise

        # Excluimos las canciones ya valoradas
        for _, row in self.ratings.iterrows():
            exclude_id = row["track_id"]
            if exclude_id in track_id_to_idx:
                final_scores[track_id_to_idx[exclude_id]] = -np.inf

        # Obtenemos las mejores recomendaciones
        top_indices = np.argsort(final_scores)[-top_n:][::-1]
        valid_indices = top_indices[final_scores[top_indices] > -np.inf]

        if len(valid_indices) == 0:
            return pd.DataFrame(columns=self.df.columns)  # ninguna recomendación

        # Devolvemos solo las columnas de interés desde el dataset original, alineadas por track_id
        recommended_track_ids = self.df_numerical.iloc[valid_indices]['track_id'].tolist()
        return self.df[self.df['track_id'].isin(recommended_track_ids)].copy()
    

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
            alpha=0.5
        )

        #plot user location
        user_pos = self.get_user_2D_position()
        if user_pos is not None:
            plt.scatter(user_pos[0], user_pos[1], color="red", s=100)

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
        plt.title(f"Like/Dislike Evolution for User {user_id}")
        plt.xlabel("Rating # (chronological)")
        plt.ylabel("Cumulative Count")
        plt.legend()
        plt.grid(True)

        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Like evolution plot saved to {filename}") 

    
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
    
    """
    """
    #Dejo los diferentes modelos para comparar cuando tengamos algún mecanismo para ello
    def get_final_recommendations(self, top_n=5, diversity=0.3):
        if len(self.ratings) == 0:
            return pd.DataFrame(columns=self.df.columns)

        df_reset = self.df.reset_index(drop=True)
        track_id_list = df_reset['track_id'].tolist()
        track_id_to_idx = {tid: i for i, tid in enumerate(track_id_list)}

        #user preference vector
        user_pref_vector = np.zeros(self.reduced_features_full.shape[1], dtype=np.float32)
        total_weight = 0.0

        for _, row in self.ratings.iterrows():
            track_id = row["track_id"]
            rating_val = row["rating"]

            if track_id not in track_id_to_idx:
                continue

            #track's embedings scaled by ratings
            tidx = track_id_to_idx[track_id]
            user_pref_vector += self.reduced_features_full[tidx] * rating_val
            total_weight += abs(rating_val)

        #if total_weight is 0, user has done something strange (skip everithing for instance)
        if total_weight == 0:
            return pd.DataFrame(columns=self.df.columns)

        #normalize the user preference vector
        user_pref_vector /= total_weight

        #we use the preference vector to query nearest neighbors
        #we will request more neighbors than top_n so we can exclude rated tracks and still have enough
        n_neighbors_to_search = max(top_n * 5, 50)
        distances, indices = self.nn_model_full.kneighbors([user_pref_vector],
                                                           n_neighbors=n_neighbors_to_search)
        distances = distances[0]
        indices = indices[0]

        #convert distances into similarities
        similarities = 1 / (1 + distances)

        #random noise for diversity
        #final_similarity = similarity*(1-diversity) + random*(diversity)
        noise = np.random.rand(len(similarities))
        final_similarities = similarities * (1 - diversity) + noise * diversity

        #we exclude songs the user has already rated
        rated_track_ids = set(self.ratings["track_id"].tolist())
        filtered_indices = []
        filtered_similarities = []

        for idx, sim in zip(indices, final_similarities):
            track_id = track_id_list[idx]
            if track_id not in rated_track_ids:
                filtered_indices.append(idx)
                filtered_similarities.append(sim)

        #if there's nothing left
        if not filtered_indices:
            return pd.DataFrame(columns=self.df.columns)

        #sort descending by similarity
        best_pairs = sorted(zip(filtered_indices, filtered_similarities),
                            key=lambda x: x[1],
                            reverse=True)

        #we take the top_n
        top_n_pairs = best_pairs[:top_n]

        final_indices = [p[0] for p in top_n_pairs]
        return df_reset.iloc[final_indices].copy()
    """
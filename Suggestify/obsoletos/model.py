import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os
import matplotlib.pyplot as plt
import psutil
import random
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors


####################ubicar e importar fichero
venv_path = os.path.dirname(os.path.dirname(sys.executable))
#conseguir direccion del proyecto, donde esta la bd
project_path = os.path.dirname(venv_path)

sqlite_conn = sqlite3.connect(project_path+"/extracted.db") #clase
#sqlite_conn = sqlite3.connect(venv_path+"/extracted.db") #portatil

cursor = sqlite_conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in SQLite:", tables)

df_og = pd.read_sql("SELECT * FROM extracted", sqlite_conn)

sqlite_conn.close()

df_og.head()


################transformar
df_og['track_id'] = df_og['track_uri'].astype('category').cat.codes
df_og['artist_id'] = df_og['artist_name'].astype('category').cat.codes
df_og['album_id'] = df_og['album_name'].astype('category').cat.codes

df = df_og[["track_id", "track_name", "artist_name", "artist_id", "album_name", "album_id", "duration_ms", "danceability", "energy", "key", "loudness", "mode", "speechiness",
                "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]]

df_todo_util = df_og[["track_id", "track_name", "artist_name", "artist_id", "album_name", "album_id", "duration_ms", "danceability", "energy", "key", "loudness", "mode", "speechiness",
                "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]]

df_others = df_og[["track_id", "track_uri", "artist_uri", "album_uri", "type", "uri", "id", "fduration_ms", "track_href", "analysis_url"]]

df_numerical = df[["track_id", "artist_id", "album_id", "duration_ms", "danceability", "energy", "key", "loudness", "mode", "speechiness",
                "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]]

df_categorical = df[["track_id", "track_name", "artist_name", "album_name"]]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numerical.drop('track_id', axis=1))


X_rank = np.linalg.matrix_rank(df_scaled)
print('Rank of X_train:', X_rank)
K_linspace = np.linspace(1, 0.75 * X_rank, 10, dtype=int)
Ks = np.unique( K_linspace)

RMSE_train = np.arange(len(Ks))




########################MODEL
UID=1
global ratings

ratings=pd.DataFrame( columns=["user", "track_id", "rating"])

rating_map = {
    "dislike": -1,
    "like": 1,
    "saltar": 0,
    "super like": 2
}


while ratings.shape[0] < 20:
    num_aleatorio = random.randint(0, len(df) - 1)
    
    #info de canción
    try:
        cancion = df.loc[num_aleatorio]
        nombre_cancion = cancion['track_name']
        artista = cancion['artist_name']
    except (KeyError, IndexError):
        nombre_cancion = "Canción desconocida"
        artista = "Artista desconocido"

    
    while True:
        input_usuario = input(f"\n🎵 Canción: {nombre_cancion} - {artista}\n""Ingrese su calificación (dislike/like/saltar/super like): ").strip().lower()        
        if input_usuario in rating_map:
            rating = rating_map[input_usuario]
            break
        else:
            print("⚠️ Entrada inválida. Ingrese una opción válida: dislike, like, saltar, super like.")

    nueva_fila = [UID, num_aleatorio, rating]
    ratings.loc[ratings.shape[0]] = nueva_fila


#######################################################################################################################################################################
# Before generating recommendations, ensure df_scaled uses the full dataset
numerical_features = ["artist_id", "album_id", "duration_ms", "danceability", "energy", "key", "loudness", "mode", "speechiness",
                "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]

# Use the full dataset for scaling and PCA
scaler = StandardScaler()
df_scaled_full = scaler.fit_transform(df_numerical[numerical_features])  # Full dataset

pca = PCA(n_components=4)
reduced_features_full = pca.fit_transform(df_scaled_full).astype(np.float32)
#######################################################################################################################################################################

# Rebuild the NearestNeighbors model with the full dataset
nn_model_full = NearestNeighbors(
    n_neighbors=50,
    metric='cosine',
    algorithm='auto',
    n_jobs=-1
).fit(reduced_features_full)

def content_based_recommendation(user_ratings, df_original, df_scaled, reduced_features, 
                               top_n=10, diversity=0.7, batch_size=100):
    """
    Sistema de recomendación basado en contenido mejorado con:
    - Manejo de dataset completo
    - Mapeo correcto de índices
    - Diversidad de recomendaciones
    """
    # 1. Preparar datos y modelos
    nn_model = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='auto', n_jobs=-1).fit(reduced_features)
    df_original_reset = df_original.reset_index()
    track_id_to_idx = {track_id: idx for idx, track_id in enumerate(df_original_reset['track_id'])}
    
    # 2. Filtrar y mapear calificaciones válidas
    valid_ratings = user_ratings[user_ratings['track_id'].isin(df_original_reset['track_id'])].copy()
    if valid_ratings.empty:
        return df_original_reset.sample(min(top_n, len(df_original_reset)))
    
    valid_ratings['track_idx'] = valid_ratings['track_id'].map(track_id_to_idx)
    valid_ratings = valid_ratings.dropna(subset=['track_idx'])
    
    # 3. Configurar pesos y exclusiones
    #rating_weights = valid_ratings['rating'].map({-2: -2.0, -1: -1.0, 0: 0.0, 1: 1.0, 2: 2.0})
    weight_map = {-1: -1.0, 0: 0.0, 1: 1.0, 2: 2.0}
    rating_weights = valid_ratings['rating'].map(weight_map)
    track_indices = valid_ratings['track_idx'].astype(int).values
    weights = rating_weights.values
    
    # 4. Búsqueda por lotes de vecinos similares
    all_scores = np.zeros(reduced_features.shape[0])
    rated_tracks = set()
    
    for i in range(0, len(track_indices), batch_size):
        batch_indices = track_indices[i:i + batch_size]
        batch_weights = weights[i:i + batch_size]
        
        distances, indices = nn_model.kneighbors(reduced_features[batch_indices])
        similarities = 1 / (1 + distances)
        weighted_scores = similarities * batch_weights[:, np.newaxis]
        
        # Actualizar puntuaciones acumuladas
        np.add.at(all_scores, indices.ravel(), weighted_scores.ravel())
        rated_tracks.update(batch_indices.tolist())
    
    # 5. Aplicar diversidad y exclusión
    noise = (np.random.rand(len(all_scores)) - 0.5) * 2 * diversity * np.mean(np.abs(all_scores))
    final_scores = all_scores * (1 - diversity) + noise
    
    # Excluir tracks ya calificados
    final_scores[list(rated_tracks)] = -np.inf
    
    # 6. Obtener mejores recomendaciones
    top_indices = np.argsort(final_scores)[-top_n:][::-1]
    valid_indices = top_indices[final_scores[top_indices] > -np.inf]
    
    if len(valid_indices) == 0:
        return pd.DataFrame(columns=df_original.columns)
    
    return df_original_reset.iloc[valid_indices].set_index('track_id')


#recomendar
recommendations = content_based_recommendation(
    user_ratings=ratings,
    df_original=df,
    df_scaled=df_scaled_full,
    reduced_features=reduced_features_full,
    top_n=5,
    diversity=0.3
)

print("Recommendations:")
print(recommendations[['track_name', 'artist_name']])







"""
backend/recommender.py
----------------------
Controlador ML (capa de recomendación).
Carga el modelo ganador B1 (PCA + Similitud del Coseno) y expone
la función principal de recomendación.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances


# Ruta base a la carpeta models/
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def cargar_recursos() -> tuple:
    """
    Carga y devuelve los tres artefactos del modelo ganador:
      - pca : transformador PCA ajustado sobre el catálogo completo.
      - catalogo_pca : matriz (174582, n_componentes) del catálogo reducido.
      - metadata : DataFrame con track_id, artist_name, track_name y popularity.

    Esta función se llama una sola vez desde app.py con @st.cache_resource,
    por lo que los artefactos permanecen en memoria durante toda la sesión.

    Devuelve
    --------
    (pca, catalogo_pca, metadata) : tupla con los tres objetos cargados.
    """
    pca = joblib.load(MODELS_DIR / "modelo_pca.joblib")
    catalogo_pca = joblib.load(MODELS_DIR / "catalogo_pca.joblib")
    metadata = joblib.load(MODELS_DIR / "catalogo_metadata.joblib")

    return pca, catalogo_pca, metadata


def recomendar(
    vector_24d: np.ndarray,
    pca,
    catalogo_pca: np.ndarray,
    df_metadata,
    catalogo_features: np.ndarray,
    track_id_entrada: str | None = None,
    k: int = 5,
) -> list[dict]:
    """
    Genera las K recomendaciones más similares a la canción de entrada.

    Parámetros
    ----------
    vector_24d : array de forma (1, 24) con las características
                 normalizadas de la canción de entrada.
    pca : transformador PCA cargado con cargar_recursos().
    catalogo_pca : catálogo completo en el espacio reducido.
    df_metadata : DataFrame con los metadatos del catálogo
                  (track_id, artist_name, track_name, popularity).
    catalogo_features : matriz completa de características sin reducir (174582, 24).
                        Se usa para extraer el vector de cada recomendación y
                        calcular el perfil acústico en la visualización.
    track_id_entrada : track_id de la canción de entrada para excluirla
                       de los resultados si pertenece al catálogo.
    k : número de recomendaciones a devolver (por defecto 5).

    Devuelve
    --------
    Lista de k dicts con claves: track_id, track_name, artist_name, score, features.
    Nota: cover_url y preview_url se enriquecen en buscador.py consultando iTunes,
    ya que no están almacenados en los metadatos del catálogo.
    """
    # 1. Proyectar el vector de entrada al espacio reducido por PCA
    vector_pca = pca.transform(vector_24d)  # (1, n_componentes)

    # 2. Calcular distancias coseno contra todo el catálogo reducido
    distancias = cosine_distances(vector_pca, catalogo_pca)[0]  # (174582,)

    # 3. Ordenar por distancia ascendente y tomar los primeros k+1
    #    (el +1 es por si la propia canción de entrada está en el catálogo)
    indices_ordenados = np.argsort(distancias)[:k + 1]

    # 4. Construir la lista de resultados excluyendo la canción de entrada
    resultados = []
    for idx in indices_ordenados:
        fila = df_metadata.iloc[idx]

        # Saltar si es la propia canción de entrada
        if track_id_entrada and fila["track_id"] == track_id_entrada:
            continue

        resultados.append({
            "track_id"   : fila["track_id"],
            "track_name" : fila["track_name"],
            "artist_name": fila["artist_name"],
            "cover_url"  : None,
            "preview_url": None,
            "score"      : round(float(1 - distancias[idx]), 4), # similitud coseno [0, 1]
            "features"   : catalogo_features[idx].tolist(),      # vector 24d para el radar
            "pca_idx"    : int(idx),                             # índice en catalogo_pca para el scatter
        })

        if len(resultados) == k:
            break

    return resultados


def obtener_vector_catalogo(
    track_id: str,
    df_metadata,
    catalogo_features: np.ndarray,
) -> np.ndarray | None:
    """
    Dado un track_id, devuelve su vector de características (24d)
    desde el catálogo local sin necesidad de procesamiento OOV.

    Parámetros
    ----------
    track_id : identificador de Spotify de la canción.
    df_metadata : DataFrame con los metadatos del catálogo
                  (track_id, artist_name, track_name, popularity).
    catalogo_features : matriz completa de características (sin reducir).

    Devuelve
    --------
    Array de forma (1, 24) o None si el track_id no está en el catálogo.
    """
    coincidencias = df_metadata.index[df_metadata["track_id"] == track_id].tolist()

    if not coincidencias:
        return None

    idx = coincidencias[0]
    return catalogo_features[idx].reshape(1, -1)
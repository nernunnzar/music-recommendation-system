"""
backend/oov_processor.py
------------------------
Controlador OOV (Out-Of-Vocabulary).
Resuelve el problema de arranque en frío procesando canciones que no
pertenecen al catálogo local: obtiene metadatos de Spotify, descarga
la muestra de audio de iTunes, extrae características acústicas mediante
DSP con Librosa y vectoriza el resultado con el pipeline de preprocesamiento.
"""

import os
import warnings
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib
import librosa
import spotipy
from dotenv import load_dotenv
from pydub import AudioSegment
from spotipy.oauth2 import SpotifyClientCredentials

warnings.filterwarnings("ignore")

# Rutas
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
TEMP_AUDIO_DIR = ROOT_DIR / "temp_audio"

TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# Cliente Spotify 
load_dotenv(ROOT_DIR / ".env")

_sp = None

def _get_spotify_client() -> spotipy.Spotify | None:
    """
    Inicializa y devuelve el cliente de Spotify de forma perezosa (lazy).
    Se crea solo la primera vez que se necesita y se reutiliza después.
    Devuelve None si las credenciales no están disponibles.
    """
    global _sp
    if _sp is not None:
        return _sp

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        return None

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
    _sp = spotipy.Spotify(auth_manager=auth_manager)
    return _sp


# Fase 1: Metadatos de Spotify
def obtener_metadatos_spotify(query: str) -> dict | None:
    """
    Busca una canción en Spotify y devuelve sus metadatos básicos.

    Parámetros
    ----------
    query : término de búsqueda libre (ej. "Bohemian Rhapsody Queen").

    Devuelve
    --------
    Dict con claves {track_id, track_name, artist_name, cover_url, duration_ms}
    o None si no se encuentra la canción o hay error de conexión.
    """
    sp = _get_spotify_client()
    if sp is None:
        return None

    try:
        resultados = sp.search(q=query, type="track", limit=1)

        if not resultados["tracks"]["items"]:
            return None

        track = resultados["tracks"]["items"][0]
        imagenes = track["album"]["images"]
        cover_url = imagenes[0]["url"] if imagenes else None

        return {
            "track_id": track["id"],
            "track_name": track["name"],
            "artist_name": track["artists"][0]["name"],
            "cover_url": cover_url,
            "duration_ms": track["duration_ms"],
        }

    except Exception:
        return None


# Fase 2: Descarga de audio (iTunes)
def descargar_audio_itunes(
    artist_name: str,
    track_name: str,
    track_id: str,
) -> Path | None:
    """
    Busca la canción en la API pública de iTunes y descarga el preview de 30s
    en formato .m4a a la carpeta temp_audio/.

    Devuelve la ruta al archivo descargado o None si no hay preview disponible.
    """
    try:
        query = urllib.parse.quote(f"{artist_name} {track_name}")
        url = (f"https://itunes.apple.com/search?"
                f"term={query}&entity=song&limit=1")
        response = requests.get(url, timeout=10)
        data = response.json()

        if data["resultCount"] == 0:
            return None

        preview_url = data["results"][0].get("previewUrl")
        if not preview_url:
            return None

        audio_data = requests.get(preview_url, timeout=15).content
        ruta_m4a = TEMP_AUDIO_DIR / f"{track_id}.m4a"

        ruta_m4a.write_bytes(audio_data)

        return ruta_m4a

    except Exception:
        return None


# Fase 3: Conversión de formato
def _convertir_a_wav(ruta_m4a: Path) -> Path | None:
    """
    Convierte un archivo .m4a a .wav usando pydub + ffmpeg.
    Devuelve la ruta al .wav o None si la conversión falla.
    """
    try:
        import shutil
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            AudioSegment.converter = ffmpeg_path
            AudioSegment.ffmpeg    = ffmpeg_path
            AudioSegment.ffprobe   = ffmpeg_path.replace("ffmpeg", "ffprobe")

        ruta_wav = ruta_m4a.with_suffix(".wav")
        AudioSegment.from_file(ruta_m4a, format="m4a").export(ruta_wav, format="wav")
        return ruta_wav
    except Exception:
        return None


# Fase 4: Extracción de características acústicas (DSP)
def _extraer_caracteristicas(ruta_wav: Path, duration_ms: float) -> dict | None:
    """
    Aplica DSP con Librosa sobre el archivo .wav y devuelve un diccionario
    con las 12 características acústicas del modelo.

    Las aproximaciones heurísticas están documentadas en detalle en la
    sección 3.3 del Capítulo IV de la memoria.
    """
    try:
        # Carga doble: raw para RMS absoluto, normalizado para análisis espectral
        y_raw, sr = librosa.load(str(ruta_wav), sr=22050)
        y = librosa.util.normalize(y_raw)

        features                = {}
        features["duration_ms"] = float(duration_ms)

        # Loudness y Energy
        rms_frames = librosa.feature.rms(y=y_raw)[0]
        rms_mean = np.mean(rms_frames)
        db_mean = np.mean(librosa.amplitude_to_db(rms_frames, ref=1.0))
        features["loudness"] = float(np.clip((db_mean + 30) / 20, 0.0, 1.0))
        p95 = np.percentile(rms_frames, 95)
        features["energy"] = float(np.clip(rms_mean / (p95 + 1e-6), 0.0, 1.0))

        # HPSS (separación armónica / percusiva)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harm_rms = np.mean(librosa.feature.rms(y=y_harmonic))
        perc_rms = np.mean(librosa.feature.rms(y=y_percussive))
        harm_ratio = harm_rms / (harm_rms + perc_rms + 1e-6)

        # Tempo
        onset_env = librosa.onset.onset_strength(
            y=y_harmonic, sr=sr, aggregate=np.median
        )
        tempo_val = float(librosa.feature.tempo(
            onset_envelope=onset_env, sr=sr
        )[0])
        if tempo_val > 145:             # corrección error de octava rítmica
            tempo_val /= 2.0
        features["tempo"] = tempo_val

        # Danceability
        bpm_score = np.exp(-((tempo_val - 120.0) ** 2) / (2 * 40.0 ** 2))
        beat_score = min(1.0, np.std(onset_env) / 2.5)
        features["danceability"] = float(bpm_score * 0.5 + beat_score * 0.5)

        # Acousticness
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, roll_percent=0.85
        )
        rolloff_norm = np.mean(rolloff) / (sr / 2)
        features["acousticness"] = float(np.clip(
            (1 - rolloff_norm) * 0.7 + harm_ratio * 0.3, 0.0, 1.0
        ))

        # Speechiness
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        vocal_var = np.mean(np.var(mfcc_delta[:4, :], axis=1))
        features["speechiness"]    = float(np.clip(vocal_var / 150.0, 0.0, 1.0))

        # Instrumentalness
        if features["speechiness"] > 0.15:
            features["instrumentalness"] = float(np.clip(
                0.05 * (1.0 - features["speechiness"]), 0.0, 1.0
            ))
        else:
            features["instrumentalness"] = float(np.clip(
                1.0 - (features["speechiness"] / 0.15), 0.0, 1.0
            ))

        # Liveness
        flatness = librosa.feature.spectral_flatness(y=y)
        features["liveness"] = float(np.mean(flatness) * 25)

        # Key y Mode (Krumhansl-Schmuckler)
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        mayor = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        menor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        notas = ["C", "C#", "D", "D#", "E", "F",
                "F#", "G", "G#", "A", "A#", "B"]
        corr_mayor = [np.corrcoef(np.roll(mayor, i), chroma_mean)[0, 1]
                      for i in range(12)]
        corr_menor = [np.corrcoef(np.roll(menor, i), chroma_mean)[0, 1]
                      for i in range(12)]
        if max(corr_mayor) >= max(corr_menor):
            features["key"] = notas[np.argmax(corr_mayor)]
            features["mode"] = "Major"
        else:
            features["key"] = notas[np.argmax(corr_menor)]
            features["mode"] = "Minor"

        # Valence
        mode_score = 0.65 if features["mode"] == "Major" else 0.35
        tempo_score = float(np.clip((tempo_val - 60) / 140, 0.0, 1.0))
        features["valence"] = float(np.clip(
            mode_score * 0.5 + features["energy"] * 0.3 + tempo_score * 0.2,
            0.0, 1.0
        ))

        # Garantizar rango [0, 1] en todas las características probabilísticas
        for col in ["energy", "acousticness", "danceability", "speechiness",
                    "valence", "instrumentalness", "liveness", "loudness"]:
            features[col] = float(np.clip(features.get(col, 0.5), 0.0, 1.0))

        return features

    except Exception:
        return None


# Fase 5: Vectorización con el pipeline entrenado
def _vectorizar(features_dict: dict) -> np.ndarray | None:
    """
    Aplica el pipeline de preprocesamiento entrenado (scaler + encoder)
    sobre el diccionario de características y devuelve el vector de 24
    dimensiones listo para el modelo.

    El uso de los transformadores entrenados sobre el dataset original es
    una restricción matemática crítica (RN-03): garantiza que el vector OOV
    sea comparable con el espacio vectorial del catálogo.
    """
    try:
        scaler = joblib.load(MODELS_DIR / "modelo_scaler.joblib")
        encoder = joblib.load(MODELS_DIR / "modelo_encoder.joblib")
        num_cols = joblib.load(MODELS_DIR / "config_num_cols.joblib")
        cat_cols = joblib.load(MODELS_DIR / "config_cat_cols.joblib")

        df_oov = pd.DataFrame([features_dict])

        # One-Hot Encoding con el encoder entrenado
        encoded_raw = encoder.transform(df_oov[cat_cols])
        encoded_cats = (encoded_raw.toarray()
                        if hasattr(encoded_raw, "toarray")
                        else encoded_raw)
        df_encoded = pd.DataFrame(
            encoded_cats,
            columns=encoder.get_feature_names_out(cat_cols),
        )

        # Escalado con el scaler entrenado + recorte de valores atípicos
        scaled_nums = scaler.transform(df_oov[num_cols])
        scaled_nums = np.clip(scaled_nums, 0.0, 1.0)
        df_scaled = pd.DataFrame(scaled_nums, columns=num_cols)

        vector = pd.concat([df_scaled, df_encoded], axis=1).values
        return vector.astype(np.float64)

    except Exception:
        return None


# Limpieza de archivos temporales
def _limpiar_audio(track_id: str) -> None:
    """Elimina los archivos .m4a y .wav temporales del track_id dado."""
    for ext in (".m4a", ".wav"):
        ruta = TEMP_AUDIO_DIR / f"{track_id}{ext}"
        if ruta.exists():
            ruta.unlink()


# Función orquestadora
def procesar_cancion_nueva(query: str) -> tuple[dict | None, np.ndarray | None]:
    """
    Función orquestadora del flujo OOV completo.
    Ejecuta secuencialmente las cinco fases y devuelve el resultado.

    Parámetros
    ----------
    query : término de búsqueda libre introducido por el usuario.

    Devuelve
    --------
    (metadatos, vector_24d) donde:
      - metadatos : dict con {track_id, track_name, artist_name, cover_url,
                    duration_ms} para renderizar la tarjeta en la interfaz.
      - vector_24d : array (1, 24) listo para el modelo de recomendación.
    Devuelve (None, None) si cualquier fase falla.
    """
    # Fase 1: metadatos de Spotify
    metadatos = obtener_metadatos_spotify(query)
    if metadatos is None:
        return None, None

    track_id = metadatos["track_id"]

    # Fase 2: descarga del preview de audio
    ruta_m4a = descargar_audio_itunes(
        metadatos["artist_name"],
        metadatos["track_name"],
        track_id,
    )
    if ruta_m4a is None:
        return None, None

    # Fase 3: conversión a .wav
    ruta_wav = _convertir_a_wav(ruta_m4a)
    if ruta_wav is None:
        _limpiar_audio(track_id)
        return None, None

    # Fase 4: extracción de características acústicas
    features = _extraer_caracteristicas(ruta_wav, metadatos["duration_ms"])
    if features is None:
        _limpiar_audio(track_id)
        return None, None

    # Fase 5: vectorización con el pipeline entrenado
    vector = _vectorizar(features)

    # Limpieza de archivos temporales independientemente del resultado
    _limpiar_audio(track_id)

    if vector is None:
        return None, None

    return metadatos, vector


# Utilidad: preview_url desde iTunes

def obtener_preview_url(artist_name: str, track_name: str) -> str | None:
    """
    Consulta la API de iTunes y devuelve únicamente el preview_url de 30s.
    Se usa en app.py para enriquecer las canciones recomendadas con audio
    sin necesidad de descargar ni procesar ningún archivo.

    Devuelve None si iTunes no dispone de preview para esa canción.
    """
    try:
        query = urllib.parse.quote(f"{artist_name} {track_name}")
        url = (f"https://itunes.apple.com/search?"
                    f"term={query}&entity=song&limit=1")
        response = requests.get(url, timeout=10)
        data = response.json()

        if data["resultCount"] == 0:
            return None

        return data["results"][0].get("previewUrl")

    except Exception:
        return None
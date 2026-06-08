"""
backend/database.py
-------------------
Modelo de datos (capa SQLite).
Gestiona la creación, escritura y lectura del historial de consultas.
Toda interacción con la base de datos pasa exclusivamente por este módulo.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


# Ruta del archivo de base de datos en la raíz del proyecto
DB_PATH = Path(__file__).resolve().parent.parent / "audyn.db"


def _conectar() -> sqlite3.Connection:
    """Abre y devuelve una conexión a la base de datos."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # permite acceder a columnas por nombre
    return conn


def inicializar_db() -> None:
    """
    Crea la tabla 'consultas' si no existe.
    Se llama una vez al arrancar la aplicación.
    """
    with _conectar() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consultas (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha           TEXT    NOT NULL,
                track_id        TEXT    NOT NULL,
                track_name      TEXT    NOT NULL,
                artist_name     TEXT    NOT NULL,
                cover_url       TEXT,
                es_oov          INTEGER NOT NULL DEFAULT 0,
                recomendaciones TEXT    NOT NULL,
                vector_entrada  TEXT
            )
        """)


def guardar_consulta(
    track_id: str,
    track_name: str,
    artist_name: str,
    cover_url: str | None,
    es_oov: bool,
    recomendaciones: list[dict],
    vector_entrada: list | None = None,
) -> None:
    """
    Inserta una nueva consulta en la base de datos.

    Parámetros
    ----------
    track_id : identificador único de Spotify de la canción de entrada.
    track_name : nombre de la canción de entrada.
    artist_name : nombre del artista de la canción de entrada.
    cover_url : URL de la carátula (puede ser None).
    es_oov : True si la canción no estaba en el catálogo local.
    recomendaciones : lista de hasta 5 dicts con claves
                      {track_name, artist_name, cover_url, preview_url}.
    vector_entrada : vector 24d de la canción de entrada (lista de floats),
                     necesario para renderizar el perfil acústico desde el historial.
    """
    fecha                = datetime.now().strftime("%d/%m/%Y %H:%M")
    recomendaciones_json = json.dumps(recomendaciones, ensure_ascii=False)
    vector_json          = json.dumps(vector_entrada, ensure_ascii=False) if vector_entrada is not None else None

    with _conectar() as conn:
        conn.execute(
            """
            INSERT INTO consultas
                (fecha, track_id, track_name, artist_name, cover_url, es_oov,
                 recomendaciones, vector_entrada)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fecha, track_id, track_name, artist_name, cover_url,
             int(es_oov), recomendaciones_json, vector_json),
        )


def consulta_ya_existe(track_id: str) -> dict | None:
    """
    Comprueba si ya existe una consulta previa para este track_id.
    Devuelve la fila más reciente como dict, o None si no existe.

    Útil para evitar reprocesar canciones OOV ya analizadas.
    """
    with _conectar() as conn:
        row = conn.execute(
            """
            SELECT * FROM consultas
            WHERE track_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (track_id,),
        ).fetchone()

    if row is None:
        return None

    return _fila_a_dict(row)


def obtener_historial() -> list[dict]:
    """
    Devuelve todas las consultas ordenadas de más reciente a más antigua.
    Cada elemento es un dict con todos los campos de la tabla,
    con 'recomendaciones' ya deserializado a lista de dicts.
    """
    with _conectar() as conn:
        rows = conn.execute(
            "SELECT * FROM consultas ORDER BY id DESC"
        ).fetchall()

    return [_fila_a_dict(row) for row in rows]


def obtener_consulta_por_id(consulta_id: int) -> dict | None:
    """
    Recupera una consulta concreta por su id.
    Devuelve None si no existe.
    """
    with _conectar() as conn:
        row = conn.execute(
            "SELECT * FROM consultas WHERE id = ?",
            (consulta_id,),
        ).fetchone()

    if row is None:
        return None

    return _fila_a_dict(row)


def _fila_a_dict(row: sqlite3.Row) -> dict:
    """
    Convierte una fila de SQLite en un diccionario Python,
    deserializando los campos JSON de recomendaciones y vector_entrada.
    """
    data = dict(row)
    data["recomendaciones"] = json.loads(data["recomendaciones"])
    data["es_oov"]          = bool(data["es_oov"])
    data["vector_entrada"]  = json.loads(data["vector_entrada"]) if data.get("vector_entrada") else None
    return data
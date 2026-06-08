"""
views/buscador.py
-----------------
Vista del buscador de Audyn.
Responsabilidades:
  · render_buscador            → renderiza el formulario de búsqueda completo.
  · enriquecer_recomendaciones → añade cover_url y preview_url a cada resultado.
  · render_recomendaciones     → pinta las 5 tarjetas en layout 3+2.
  · _procesar_y_guardar        → orquesta cover, cálculo ML y session_state.

La lógica de datos (ML, OOV, BD) se delega íntegramente al backend.
"""

import streamlit as st
import numpy as np
import pandas as pd

from backend.database      import guardar_consulta, consulta_ya_existe
from backend.recommender   import recomendar, obtener_vector_catalogo
from backend.oov_processor import (
    procesar_cancion_nueva,
    obtener_metadatos_spotify,
    obtener_preview_url,
)
from views.components import render_cancion_entrada, render_tarjeta, render_perfil_acustico


# Enriquecimiento de la canción de entrada
def _enriquecer_entrada(metadatos: dict) -> dict:
    """
    Añade cover_url a la canción de entrada si no viene ya informado,
    consultando Spotify por nombre + artista.
    """
    if not metadatos.get("cover_url"):
        meta = obtener_metadatos_spotify(
            f"{metadatos['track_name']} {metadatos['artist_name']}"
        )
        if meta:
            metadatos["cover_url"] = meta.get("cover_url")
    return metadatos


# Enriquecimiento de recomendaciones
def enriquecer_recomendaciones(recomendaciones: list[dict]) -> list[dict]:
    """
    Añade cover_url (vía Spotify) y preview_url (vía iTunes) a cada
    recomendación que no los tenga ya almacenados.

    Devuelve una nueva lista de dicts enriquecidos.
    """
    enriquecidas = []
    for rec in recomendaciones:
        rec = dict(rec)

        if not rec.get("cover_url"):
            meta = obtener_metadatos_spotify(
                f"{rec['track_name']} {rec['artist_name']}"
            )
            if meta:
                rec["cover_url"] = meta.get("cover_url")

        if not rec.get("preview_url"):
            rec["preview_url"] = obtener_preview_url(
                rec["artist_name"], rec["track_name"]
            )

        enriquecidas.append(rec)

    return enriquecidas


# Preparación de metadatos y vector según origen
def _preparar_desde_catalogo(
    seleccion: str,
    mapa_trackid: dict,
    df_metadata: pd.DataFrame,
    catalogo_features: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """
    Construye los metadatos y vector de entrada para una canción del catálogo local.
    """
    track_id = mapa_trackid[seleccion]
    fila     = df_metadata[df_metadata["track_id"] == track_id].iloc[0]

    metadatos = {
        "track_id"   : track_id,
        "track_name" : fila["track_name"],
        "artist_name": fila["artist_name"],
        "cover_url"  : None,
        "es_oov"     : False,
    }
    vector = obtener_vector_catalogo(track_id, df_metadata, catalogo_features)
    return metadatos, vector


def _preparar_desde_oov(texto_oov: str) -> tuple[dict | None, np.ndarray | None]:
    """
    Procesa una canción OOV (fuera del catálogo) vía Spotify + iTunes + DSP.
    Devuelve (None, None) si el procesamiento falla.
    """
    with st.spinner("Analizando canción, esto puede tardar unos segundos..."):
        metadatos, vector = procesar_cancion_nueva(texto_oov)

    if metadatos is not None:
        metadatos["es_oov"] = True

    return metadatos, vector


# Renderizado de recomendaciones
def render_recomendaciones(recomendaciones: list[dict]) -> None:
    """
    Renderiza las 5 canciones recomendadas en una única fila.
    Espera recibir las recomendaciones ya enriquecidas con
    cover_url y preview_url.
    """
    st.markdown(
        "<p style='font-size:1.1rem; font-weight:700; color:#0d1b2a; "
        "margin-bottom:1rem;'>Recomendaciones</p>",
        unsafe_allow_html=True,
    )

    # Fila única: las 5 canciones en una sola fila
    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i < len(recomendaciones):
            render_tarjeta(col, recomendaciones[i])


# Orquestador interno
def _procesar_y_guardar(
    metadatos: dict,
    vector: np.ndarray,
    consulta_previa: dict | None,
    pca,
    catalogo_pca: np.ndarray,
    df_metadata: pd.DataFrame,
    catalogo_features: np.ndarray,
) -> None:
    """
    Función auxiliar que:
      1. Obtiene cover_url de la canción de entrada si falta.
      2. Reutiliza recomendaciones cacheadas en BD o las calcula con el modelo.
      3. Enriquece las recomendaciones con cover_url y preview_url.
      4. Persiste la consulta en SQLite si es nueva (INSERT OR IGNORE).
      5. Actualiza session_state y fuerza el rerun.
    """
    # 1. Cover de la canción de entrada
    metadatos = _enriquecer_entrada(metadatos)
    st.session_state["cancion_entrada"] = metadatos

    # 2. Recomendaciones: caché BD o cálculo nuevo
    if consulta_previa:
        recomendaciones = consulta_previa["recomendaciones"]
        vector_flat     = consulta_previa.get("vector_entrada")
    else:
        with st.spinner("Calculando recomendaciones..."):
            recomendaciones = recomendar(
                vector,
                pca,
                catalogo_pca,
                df_metadata,
                catalogo_features,
                track_id_entrada=metadatos["track_id"],
            )

        # 3. Enriquecer con cover_url y preview_url
        recomendaciones = enriquecer_recomendaciones(recomendaciones)
        vector_flat     = vector.flatten().tolist()

        # 4. Persistir en SQLite (INSERT OR IGNORE evita duplicados)
        guardar_consulta(
            track_id        = metadatos["track_id"],
            track_name      = metadatos["track_name"],
            artist_name     = metadatos["artist_name"],
            cover_url       = metadatos.get("cover_url"),
            es_oov          = metadatos.get("es_oov", False),
            recomendaciones = recomendaciones,
            vector_entrada  = vector_flat,
        )

    # 5. Guardar vector de entrada y actualizar estado
    st.session_state["vector_entrada"]   = vector_flat
    st.session_state["recomendaciones"]  = recomendaciones
    st.session_state["limpiar_buscador"] = True
    st.rerun()


# Vista principal
def render_buscador(
    pca,
    catalogo_pca: np.ndarray,
    df_metadata: pd.DataFrame,
    catalogo_features: np.ndarray,
    opciones: list[str],
    mapa_trackid: dict,
) -> None:
    """
    Renderiza la vista completa del buscador.

    Recibe los recursos cacheados desde app.py para no depender
    de llamadas a st.cache_* dentro de la vista.
    """
    st.markdown(
        "<p class='page-title'>Descubre nueva música</p>",
        unsafe_allow_html=True,
    )

    # Limpiar campos si viene de una búsqueda anterior
    if st.session_state["limpiar_buscador"]:
        st.session_state["limpiar_buscador"] = False
        st.session_state["sel_catalogo"] = None
        st.session_state["texto_oov"] = ""

    # Cargar resultados precargados desde el historial
    if st.session_state["resultado_historial"] is not None:
        datos = st.session_state["resultado_historial"]
        st.session_state["cancion_entrada"] = datos["cancion_entrada"]
        st.session_state["recomendaciones"] = datos["recomendaciones"]
        st.session_state["vector_entrada"]  = datos.get("vector_entrada")
        st.session_state["resultado_historial"] = None


    # Sección A: Catálogo local
    st.markdown(
        "<p style='font-size:0.85rem; color:#6b7280; margin-bottom:0.4rem;'>"
        "Busca en el catálogo</p>",
        unsafe_allow_html=True,
    )

    seleccion_catalogo = st.selectbox(
        label = "catalogo",
        options = opciones,
        index = None,
        placeholder = "Escribe el nombre de una canción o artista...",
        label_visibility = "collapsed",
        key = "sel_catalogo",
    )


    # Sección B: Spotify (OOV)
    st.markdown(
        "<div style='margin-top:1.25rem'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.85rem; color:#6b7280; margin-bottom:0.4rem;'>"
        "¿No encuentras tu canción? Búscala en Spotify</p>",
        unsafe_allow_html=True,
    )

    texto_oov = st.text_input(
        label = "spotify",
        placeholder = "Escribe el nombre de la canción y el artista...",
        label_visibility = "collapsed",
        key = "texto_oov",
    )

    buscar = st.button(
        "Generar recomendaciones",
        key = "btn_buscar",
        use_container_width = False,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Lógica: búsqueda en catálogo
    if buscar and seleccion_catalogo:
        metadatos, vector = _preparar_desde_catalogo(
            seleccion_catalogo, mapa_trackid, df_metadata, catalogo_features
        )
        consulta_previa = consulta_ya_existe(metadatos["track_id"])

        _procesar_y_guardar(
            metadatos, vector, consulta_previa,
            pca, catalogo_pca, df_metadata, catalogo_features,
        )

    if buscar and texto_oov and not seleccion_catalogo:
        metadatos, vector = _preparar_desde_oov(texto_oov)

        if metadatos is None or vector is None:
            st.error(
                "No se ha podido procesar la canción. "
                "Comprueba tu conexión o intenta con otro título."
            )
            return

        consulta_previa = consulta_ya_existe(metadatos["track_id"])

        _procesar_y_guardar(
            metadatos, vector, consulta_previa,
            pca, catalogo_pca, df_metadata, catalogo_features,
        )

    # Mostrar resultados
    if st.session_state["cancion_entrada"]:
        render_cancion_entrada(st.session_state["cancion_entrada"])

    if st.session_state["recomendaciones"]:
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        render_recomendaciones(st.session_state["recomendaciones"])
        render_perfil_acustico(
            st.session_state["cancion_entrada"],
            st.session_state["recomendaciones"],
            st.session_state["vector_entrada"],
        )
"""
app.py
------
Punto de entrada de Audyn — Sistema de Recomendación Musical.
Arquitectura MVC adaptada a Streamlit:
  · Vista      → views/buscador.py, views/historial.py, views/components.py
  · Controlador → backend/recommender.py, backend/oov_processor.py
  · Modelo     → backend/database.py (SQLite)

Este archivo es exclusivamente responsable de:
  1. Configuración de página y CSS global.
  2. Inicialización de BD y carga de recursos en caché.
  3. Inicialización del session_state.
  4. Renderizado del sidebar de navegación.
  5. Enrutado hacia la vista activa.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from backend.database    import inicializar_db
from backend.recommender import cargar_recursos
from views.buscador      import render_buscador
from views.historial     import render_historial


# Configuración de página
st.set_page_config(
    page_title = "Audyn",
    page_icon  = "assets/logo.png",
    layout     = "wide",
)

# Deshabilitar colapso del sidebar
st.markdown("""
    <style>
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)


# CSS externo
def _cargar_css() -> None:
    """Inyecta el archivo CSS externo en la aplicación."""
    ruta = Path(__file__).resolve().parent / "assets" / "style.css"
    with open(ruta, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_cargar_css()


# Inicialización única de la base de datos
inicializar_db()


# Recursos en caché
@st.cache_resource
def _cargar_modelo():
    """
    Carga y cachea el modelo PCA, el catálogo reducido y los metadatos.
    Se ejecuta una única vez al arrancar la app.
    """
    return cargar_recursos()


@st.cache_data
def _cargar_catalogo_features():
    """
    Carga y cachea el dataset procesado completo (174.582 × 24 dimensiones).
    Se usa para obtener el vector de canciones que ya están en el catálogo,
    evitando lanzar el flujo OOV innecesariamente.
    """
    ruta = Path(__file__).resolve().parent / "data" / "dataset_procesado.csv"
    df   = pd.read_csv(ruta)

    metadata_cols = ["track_id", "artist_name", "track_name", "popularity"]
    feature_cols  = [c for c in df.columns if c not in metadata_cols]

    return df[metadata_cols].reset_index(drop=True), df[feature_cols].values


@st.cache_data
def _cargar_opciones_buscador():
    """
    Genera y cachea la lista de opciones para el autocompletado del buscador.
    Cada opción tiene el formato 'Nombre de canción — Artista'.
    Devuelve también un dict de búsqueda rápida por esa cadena.
    Solo carga las 5.000 canciones más populares para mejorar el rendimiento.
    """
    ruta = Path(__file__).resolve().parent / "data" / "dataset_procesado.csv"
    df   = pd.read_csv(
        ruta, usecols=["track_id", "track_name", "artist_name", "popularity"]
    )

    df = df.dropna(subset=["track_name", "artist_name"])
    df["track_name"]  = df["track_name"].astype(str)
    df["artist_name"] = df["artist_name"].astype(str)
    df = df.sort_values("popularity", ascending=False).head(5000)

    df["etiqueta"] = df["track_name"] + " — " + df["artist_name"]
    opciones       = df["etiqueta"].tolist()
    mapa_trackid   = dict(zip(df["etiqueta"], df["track_id"]))

    return opciones, mapa_trackid


# Session state: valores por defecto
_DEFAULTS = {
    "pagina"             : "buscador",
    "resultado_historial": None,
    "cancion_entrada"    : None,
    "recomendaciones"    : None,
    "limpiar_buscador"   : False,
}
for clave, valor in _DEFAULTS.items():
    if clave not in st.session_state:
        st.session_state[clave] = valor


# Sidebar
def _render_sidebar() -> None:
    """Renderiza el sidebar con logo y botones de navegación."""
    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)
        st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

        for pagina, icono_svg, etiqueta, key in [
            (
                "buscador",
                """<circle cx="11" cy="11" r="8"/>
                   <line x1="21" y1="21" x2="16.65" y2="16.65"/>""",
                "Buscador",
                "btn_nav_buscador",
            ),
            (
                "historial",
                """<circle cx="12" cy="12" r="10"/>
                   <polyline points="12 6 12 12 16 14"/>""",
                "Historial",
                "btn_nav_historial",
            ),
        ]:
            activo = st.session_state["pagina"] == pagina
            st.markdown(f"""
                <div class='nav-btn {"active" if activo else ""}'>
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        {icono_svg}
                    </svg>
                    <span>{etiqueta}</span>
                </div>
            """, unsafe_allow_html=True)

            if st.button(etiqueta, key=key, use_container_width=True):
                st.session_state["pagina"]              = pagina
                st.session_state["resultado_historial"] = None
                st.rerun()


# Enrutador principal
_render_sidebar()

if st.session_state["pagina"] == "buscador":
    pca, catalogo_pca, metadata = _cargar_modelo()
    df_metadata, catalogo_features = _cargar_catalogo_features()
    opciones, mapa_trackid = _cargar_opciones_buscador()

    render_buscador(
        pca               = pca,
        catalogo_pca      = catalogo_pca,
        df_metadata       = df_metadata,
        catalogo_features = catalogo_features,
        opciones          = opciones,
        mapa_trackid      = mapa_trackid,
    )

elif st.session_state["pagina"] == "historial":
    render_historial()
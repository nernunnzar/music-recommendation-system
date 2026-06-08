"""
views/historial.py
------------------
Vista del historial de búsquedas de Audyn.
Responsabilidades:
  · render_historial → lista paginada de consultas pasadas con botón
                       para recuperar las recomendaciones en el buscador.
"""

import streamlit as st

from backend.database import obtener_historial


def render_historial() -> None:
    """Renderiza la vista del historial de recomendaciones."""

    st.markdown(
        "<p class='page-title'>Historial de recomendaciones</p>",
        unsafe_allow_html=True,
    )

    historial = obtener_historial()

    if not historial:
        st.markdown("""
            <div class='historial-container'>
                <div class='empty-state'>
                    <div class='empty-icon'>🎵</div>
                    <p>Aún no hay búsquedas registradas.<br>
                    ¡Busca una canción para empezar!</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        return

    # Inyectar estilo scoped para los botones de esta vista únicamente
    st.markdown("""
        <style>
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] .stButton > button {
            background-color: transparent !important;
            color           : #3b6ef8 !important;
            border          : 2px solid #3b6ef8 !important;
            box-shadow      : none !important;
        }
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] .stButton > button:hover {
            background-color: #3b6ef8 !important;
            color           : #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Cabecera de la tabla
    _ESTILO_CABECERA = (
        "font-size:0.8rem; font-weight:700; color:#6b7280; "
        "text-transform:uppercase; letter-spacing:0.05em; "
        "padding:0.75rem 0; margin:0; border-bottom: 2px solid #e5e7eb;"
    )

    col_fecha, col_cancion, col_badge, col_btn = st.columns([2, 4, 1.5, 1.5])
    with col_fecha:
        st.markdown(f"<p style='{_ESTILO_CABECERA}'>Fecha</p>",           unsafe_allow_html=True)
    with col_cancion:
        st.markdown(f"<p style='{_ESTILO_CABECERA}'>Canción buscada</p>", unsafe_allow_html=True)
    with col_badge:
        st.markdown(f"<p style='{_ESTILO_CABECERA} text-align:center;'>Origen</p>", unsafe_allow_html=True)
    with col_btn:
        st.markdown(f"<p style='{_ESTILO_CABECERA}'>Recomendaciones</p>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

    # Filas del historial
    for consulta in historial:
        partes = consulta["fecha"].split(" ")
        fecha  = partes[0] if partes else consulta["fecha"]
        hora   = partes[1] if len(partes) > 1 else ""

        col_fecha, col_cancion, col_badge, col_btn = st.columns([2, 4, 1.5, 1.5])

        with col_fecha:
            st.markdown(f"""
                <div class='historial-fecha'>
                    {fecha}<span>{hora}</span>
                </div>
            """, unsafe_allow_html=True)

        with col_cancion:
            st.markdown(f"""
                <div class='historial-cancion'>
                    {consulta['track_name']} — {consulta['artist_name']}
                </div>
            """, unsafe_allow_html=True)

        with col_badge:
            es_oov      = consulta.get("es_oov", False)
            badge_text  = "Spotify / OOV" if es_oov else "Catálogo"
            badge_color = "#f97316"        if es_oov else "#3b6ef8"
            st.markdown(f"""
                <div style='padding-top:0.3rem; text-align:center;'>
                    <span style='
                        background-color: {badge_color};
                        color: #ffffff;
                        font-size: 0.7rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        padding: 0.2rem 0.6rem;
                        border-radius: 20px;
                        white-space: nowrap;
                    '>{badge_text}</span>
                </div>
            """, unsafe_allow_html=True)

        with col_btn:
            if st.button(
                "Ver resultados",
                key  = f"btn_historial_{consulta['id']}",
            ):
                st.session_state["resultado_historial"] = {
                    "cancion_entrada": {
                        "track_id"   : consulta["track_id"],
                        "track_name" : consulta["track_name"],
                        "artist_name": consulta["artist_name"],
                        "cover_url"  : consulta.get("cover_url"),
                        "es_oov"     : consulta.get("es_oov", False),
                    },
                    "recomendaciones": consulta["recomendaciones"],
                    "vector_entrada" : consulta.get("vector_entrada"),
                }
                st.session_state["pagina"]          = "buscador"
                st.session_state["cancion_entrada"] = None
                st.session_state["recomendaciones"] = None
                st.rerun()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
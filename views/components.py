"""
views/components.py
-------------------
Componentes visuales reutilizables de Audyn.
Contiene los bloques HTML/Streamlit más atómicos:
  · render_cancion_entrada → tarjeta de la canción buscada.
  · _render_tarjeta → tarjeta individual de recomendación.
"""

import streamlit as st


def render_cancion_entrada(metadatos: dict) -> None:
    """Renderiza la tarjeta de la canción de entrada."""
    cover = metadatos.get("cover_url") or "https://via.placeholder.com/80"
    st.markdown(f"""
        <div class='entrada-card'>
            <img src='{cover}' alt='caratula'/>
            <div class='entrada-info'>
                <div class='entrada-label'>Canción seleccionada</div>
                <div class='entrada-title'>{metadatos['track_name']}</div>
                <div class='entrada-artist'>{metadatos['artist_name']}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_tarjeta(col, rec: dict) -> None:
    """Renderiza una tarjeta individual de canción recomendada."""
    with col:
        cover = rec.get("cover_url") or "https://via.placeholder.com/300"
        preview = rec.get("preview_url")

        no_preview_html = "" if preview else """
            <div style='font-size:0.8rem; color:#9ca3af;
                        padding:0.5rem 0; text-align:center;'>
                Preview no disponible
            </div>
        """

        st.markdown(f"""
            <div class='song-card'>
                <img src='{cover}' alt='caratula'/>
                <div class='song-title'>{rec['track_name']}</div>
                <div class='song-artist'>{rec['artist_name']}</div>
                {no_preview_html}
            </div>
        """, unsafe_allow_html=True)

        # Reproductor fuera del HTML pero visualmente integrado
        if preview:
            st.audio(preview, format="audio/mp4")
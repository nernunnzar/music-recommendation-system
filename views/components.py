"""
views/components.py
-------------------
Componentes visuales reutilizables de Audyn.
Contiene los bloques HTML/Streamlit más atómicos:
  · render_cancion_entrada  → tarjeta de la canción buscada.
  · render_tarjeta          → tarjeta individual de recomendación.
  · render_perfil_acustico  → radar comparativo + barras de similitud.
"""

import streamlit as st
import plotly.graph_objects as go


def render_cancion_entrada(metadatos: dict) -> None:
    """Renderiza la tarjeta de la canción de entrada."""
    cover   = metadatos.get("cover_url") or "https://via.placeholder.com/80"
    es_oov  = metadatos.get("es_oov", False)

    badge_text  = "Spotify / OOV" if es_oov else "Catálogo"
    badge_color = "#f97316"        if es_oov else "#3b6ef8"

    st.markdown(f"""
        <div class='entrada-card'>
            <img src='{cover}' alt='caratula'/>
            <div class='entrada-info'>
                <div class='entrada-label'>Canción seleccionada</div>
                <div class='entrada-title'>{metadatos['track_name']}</div>
                <div class='entrada-artist'>{metadatos['artist_name']}</div>
                <div style='margin-top:0.5rem;'>
                    <span style='
                        background-color: {badge_color};
                        color: #ffffff;
                        font-size: 0.7rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        padding: 0.2rem 0.6rem;
                        border-radius: 20px;
                    '>{badge_text}</span>
                </div>
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


# Índices de las 6 métricas seleccionadas dentro del vector de 24 dimensiones.
# Orden de num_cols: acousticness(0), danceability(1), duration_ms(2),
#                   energy(3), instrumentalness(4), liveness(5),
#                   loudness(6), speechiness(7), tempo(8), valence(9)
# Se excluyen: duration_ms, instrumentalness, speechiness y tempo.
_METRICAS_RADAR = {
    "Acousticness" : 0,
    "Danceability" : 1,
    "Energy"       : 3,
    "Liveness"     : 5,
    "Loudness"     : 6,
    "Valence"      : 9,
}

# Longitud máxima de nombres en gráficos (el hover siempre muestra el completo)
_MAX_CHARS_RADAR  = 40
_MAX_CHARS_BARRAS = 28

def _truncar(texto: str, max_chars: int) -> str:
    """Trunca el texto a max_chars caracteres añadiendo '…' si es necesario."""
    return texto if len(texto) <= max_chars else texto[:max_chars].rstrip() + "…"


# Paleta: entrada en azul corporativo, recomendaciones en colores pastel
_COLOR_ENTRADA = "#3b6ef8"
_COLORES_RECS  = [
    "rgba(251, 178, 133, 0.75)",   # naranja pastel
    "rgba(134, 213, 152, 0.75)",   # verde pastel
    "rgba(206, 166, 251, 0.75)",   # morado pastel
    "rgba(249, 168, 202, 0.75)",   # rosa pastel
    "rgba(129, 222, 214, 0.75)",   # teal pastel
]
_COLORES_RECS_LINE = [
    "rgba(234, 120,  50, 0.90)",   # naranja pastel oscuro
    "rgba(60,  180,  90, 0.90)",   # verde pastel oscuro
    "rgba(160,  90, 240, 0.90)",   # morado pastel oscuro
    "rgba(230,  90, 160, 0.90)",   # rosa pastel oscuro
    "rgba(30,  180, 170, 0.90)",   # teal pastel oscuro
]


def render_perfil_acustico(
    cancion_entrada: dict,
    recomendaciones: list[dict],
    vector_entrada: list,
) -> None:
    """
    Renderiza el perfil acústico comparativo: radar + barras de similitud.

    Parámetros
    ----------
    cancion_entrada : dict con track_name y artist_name de la canción buscada.
    recomendaciones : lista de dicts enriquecidos; cada uno debe tener
                      'features' (vector 24d) y 'score' (similitud coseno).
    vector_entrada  : vector 24d (list o np.ndarray) de la canción de entrada,
                      almacenado en session_state['vector_entrada'].
    """
    # Filtrar solo recomendaciones que llegaron con features
    recs_con_features = [r for r in recomendaciones if r.get("features")]
    if not recs_con_features or vector_entrada is None:
        return

    etiquetas = list(_METRICAS_RADAR.keys())
    indices   = list(_METRICAS_RADAR.values())

    # Valores de la canción de entrada
    vals_entrada = [float(vector_entrada[i]) for i in indices]

    st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:1.1rem; font-weight:700; color:#0d1b2a; "
        "margin-bottom:1rem;'>Perfil acústico</p>",
        unsafe_allow_html=True,
    )

    col_radar, col_barras = st.columns([3, 2], gap="large")

    # Columna izquierda: Radar
    with col_radar:
        st.markdown(
            "<p style='font-size:0.9rem; font-weight:700; color:#0d1b2a; "
            "margin-bottom:0.5rem;'>Características acústicas comparadas</p>",
            unsafe_allow_html=True,
        )

        fig_radar = go.Figure()

        for i, rec in enumerate(recs_con_features):
            vals_rec     = [float(rec["features"][j]) for j in indices]
            nombre       = rec["track_name"]
            nombre_corto = _truncar(nombre, _MAX_CHARS_RADAR)

            fig_radar.add_trace(go.Scatterpolar(
                r         = vals_rec + [vals_rec[0]],
                theta     = etiquetas + [etiquetas[0]],
                fill      = "toself",
                name      = nombre_corto,
                line      = dict(color=_COLORES_RECS_LINE[i], width=1.5),
                fillcolor = _COLORES_RECS[i],
                hovertemplate = (
                    f"<b>{nombre}</b><br>"
                    "%{theta}: %{r:.2f}<extra></extra>"
                ),
            ))

        # Canción de entrada encima, sin relleno para no tapar las demás
        fig_radar.add_trace(go.Scatterpolar(
            r         = vals_entrada + [vals_entrada[0]],
            theta     = etiquetas + [etiquetas[0]],
            fill      = "none",
            name      = _truncar(cancion_entrada["track_name"], _MAX_CHARS_RADAR),
            line      = dict(color=_COLOR_ENTRADA, width=3),
            hovertemplate = (
                f"<b>{cancion_entrada['track_name']}</b><br>"
                "%{theta}: %{r:.2f}<extra></extra>"
            ),
        ))

        fig_radar.update_layout(
            polar = dict(
                bgcolor    = "#f8f9fb",
                radialaxis = dict(
                    visible  = True,
                    range    = [0, 1],
                    tickfont = dict(size=9, color="#9ca3af"),
                    gridcolor= "#e5e7eb",
                    linecolor= "#e5e7eb",
                ),
                angularaxis = dict(
                    tickfont  = dict(size=11, color="#0d1b2a", family="Segoe UI"),
                    gridcolor = "#e5e7eb",
                    linecolor = "#e5e7eb",
                ),
            ),
            showlegend  = True,
            legend      = dict(
                orientation    = "h",
                yanchor        = "bottom",
                y              = -0.45,
                xanchor        = "center",
                x              = 0.5,
                font           = dict(size=10, color="#0d1b2a"),
                entrywidth     = 0.33,
                entrywidthmode = "fraction",
            ),
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            margin        = dict(t=20, b=100, l=40, r=40),
            height        = 450,
        )

        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    # Columna derecha: Barras de similitud
    with col_barras:
        st.markdown(
            "<p style='font-size:0.9rem; font-weight:700; color:#0d1b2a; "
            "margin-bottom:0.5rem;'>Similitud acústica</p>",
            unsafe_allow_html=True,
        )

        nombres        = [r["track_name"] for r in recs_con_features]
        nombres_cortos = [_truncar(n, _MAX_CHARS_BARRAS) for n in nombres]
        scores         = [round(r.get("score", 0), 4) for r in recs_con_features]
        colores_barras = [c.replace("0.75", "0.90") for c in _COLORES_RECS]

        fig_barras = go.Figure()
        fig_barras.add_trace(go.Bar(
            x            = scores,
            y            = nombres_cortos,
            orientation  = "h",
            width        = 0.25,
            marker       = dict(color=colores_barras[:len(nombres)]),
            text         = [f"{s:.5f}" for s in scores],
            textposition = "outside",
            textfont     = dict(size=14, color="#0d1b2a"),
            customdata   = nombres,
            hovertemplate= "<b>%{customdata}</b><br>Similitud: %{x:.5f}<extra></extra>",
        ))

        # Rango dinámico: zoom sobre la zona donde están los scores
        margen = (max(scores) - min(scores)) * 0.5 if len(scores) > 1 else 0.001
        x_min  = max(0, min(scores) - margen)
        x_max  = min(1, max(scores) + margen * 3)

        fig_barras.update_layout(
            xaxis = dict(
                range      = [x_min, x_max],
                visible    = False,
                fixedrange = True,
            ),
            yaxis = dict(
                autorange  = "reversed",
                tickfont   = dict(size=14, color="#0d1b2a"),
                showgrid   = False,
                zeroline   = False,
                fixedrange = True,
            ),
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            margin        = dict(t=10, b=10, l=10, r=110),
            height        = 450,
            showlegend    = False,
            dragmode      = False,
        )

        st.plotly_chart(fig_barras, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
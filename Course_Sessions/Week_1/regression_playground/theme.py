"""
theme.py
========
The shared visual identity for the Regression Playground, lifted directly from
the Week 2 lecture slides so the app reads like a live slide.

Palette, typography and Plotly styling all live here — change a hex code once
and the whole app (and every chart) updates.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Palette  (identical hex codes to the course slide deck)
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#F4F1EA",        # warm cream page background
    "card": "#ECE7DC",      # slightly darker cream — panels/cards
    "ink": "#171A23",       # near-black — titles & primary text
    "ink_soft": "#5A6273",  # slate grey — subtitles, captions, axis text
    "rule": "#D8D2C4",      # light taupe — hairlines, borders, gridlines
    "data": "#1C3D5A",      # deep navy — data points / training
    "data_edge": "#0E2239", # darker navy — marker edges
    "model": "#A8622D",     # terracotta — the model / fitted line / accent
    "error": "#7A2E1E",     # brick red — errors, residuals, danger
    "good": "#3E7063",      # muted teal — the good outcome (minimum, test, "just right")
    "neutral": "#6B4E71",   # muted plum — a neutral fourth category
    "grid": "#E4DFD1",      # pale gridline
}

# Semantic aliases used across the app
DATA = PALETTE["data"]
MODEL = PALETTE["model"]
ERROR = PALETTE["error"]
GOOD = PALETTE["good"]
NEUTRAL = PALETTE["neutral"]
INK = PALETTE["ink"]
INK_SOFT = PALETTE["ink_soft"]

# Metric accent bars (match slide_24_regression_metrics)
METRIC_ACCENTS = {
    "MAE": "#A8622D",
    "MSE": "#7A2E1E",
    "RMSE": "#6B4E71",
    "R2": "#3E7063",
    "MAPE": "#1C3D5A",
}


# ---------------------------------------------------------------------------
# Page-level CSS
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Apply the cream/terracotta slide look to the Streamlit page."""
    p = PALETTE
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {p['bg']}; }}
        /* Headings use a serif face, like the slide titles */
        h1, h2, h3, h4 {{
            font-family: Georgia, 'Times New Roman', serif !important;
            color: {p['ink']};
        }}
        .rp-header {{
            padding: 1.0rem 1.3rem; border-radius: 4px;
            border-left: 6px solid {p['model']};
            background: {p['card']};
            margin-bottom: 1.0rem;
        }}
        .rp-title {{
            font-family: Georgia, serif; font-size: 1.85rem; font-weight: 700;
            color: {p['ink']}; margin: 0;
        }}
        .rp-sub {{ color: {p['ink_soft']}; font-size: 0.95rem; font-style: italic;
                   margin-top: 0.2rem; }}
        /* Challenge / reveal call-out boxes */
        .rp-challenge {{
            background: #F6EFE4; border: 1px solid {p['rule']};
            border-left: 5px solid {p['model']};
            padding: 0.7rem 1rem; border-radius: 4px; margin: 0.4rem 0 0.8rem 0;
            color: {p['ink']};
        }}
        .rp-note {{ color: {p['ink_soft']}; font-size: 0.85rem; }}
        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: {p['card']}; border: 1px solid {p['rule']};
            border-top: 3px solid {p['model']};
            padding: 12px 14px; border-radius: 4px;
        }}
        div[data-testid="stMetric"] label p {{ color: {p['ink_soft']} !important;
            font-weight: 600; }}
        div[data-testid="stMetricValue"] {{ color: {p['ink']}; }}
        section[data-testid="stSidebar"] {{ background: {p['card']};
            border-right: 1px solid {p['rule']}; }}
        .rp-footer {{ color: {p['ink_soft']}; font-size: 0.8rem;
            border-top: 1px solid {p['rule']}; padding-top: 0.5rem; margin-top: 1.2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="rp-header">
            <p class="rp-title">{title}</p>
            <div class="rp-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def challenge(text: str) -> None:
    """Render a guided-mode challenge prompt."""
    st.markdown(f'<div class="rp-challenge"><b>Challenge&nbsp;·</b> {text}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------
def style_fig(fig, height: int = 360, title: str | None = None):
    """Apply the slide look to a Plotly figure (cream background, serif title).

    Note: the title text is always a string ("" when absent) - passing None makes
    plotly.js render the literal word "undefined" in the browser.
    """
    p = PALETTE
    fig.update_layout(
        height=height,
        title=dict(text=title or "", font=dict(family="Georgia, serif", size=16,
                                               color=p["ink"])),
        margin=dict(l=10, r=10, t=40 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Helvetica Neue, Arial, sans-serif", color=p["ink_soft"], size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=p["rule"], borderwidth=0),
        colorway=[DATA, MODEL, GOOD, NEUTRAL, ERROR],
    )
    fig.update_xaxes(gridcolor=p["grid"], zerolinecolor=p["rule"],
                     linecolor=p["rule"], tickcolor=p["rule"])
    fig.update_yaxes(gridcolor=p["grid"], zerolinecolor=p["rule"],
                     linecolor=p["rule"], tickcolor=p["rule"])
    return fig

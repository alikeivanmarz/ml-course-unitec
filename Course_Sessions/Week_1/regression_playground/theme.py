"""Shared visual identity for the Regression Playground.

The warm cream/terracotta identity comes from the course slides.  Normal body
copy uses a sans-serif face for readability, while headings retain the slide
deck's Georgia/serif character.  UI accents and chart series have separate
tokens: the decorative terracotta can stay warm, while interactive states use
a darker colour that meets WCAG AA contrast on both page and card backgrounds.
"""

from __future__ import annotations

from html import escape

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#F4F1EA",          # warm cream page background
    "card": "#ECE7DC",        # slightly darker cream — panels/cards
    "ink": "#171A23",         # near-black — titles and primary text
    "ink_soft": "#5A6273",    # slate grey — subtitles, captions, axis text
    "rule": "#D8D2C4",        # light taupe — hairlines, borders, gridlines
    "grid": "#E4DFD1",        # pale gridline
    "terracotta": "#A8622D",  # slide-deck accent for decorative, non-text use
    "accent": "#934C1F",      # AA on bg (5.65:1) and card (5.17:1)
    "data": "#1C3D5A",        # deep navy — observations
    "data_edge": "#0E2239",   # darker navy — marker edges
    "model": "#934C1F",       # fitted line / accessible terracotta
    "error": "#7A2E1E",       # brick red — errors, residuals, danger
    "good": "#3E7063",        # muted teal — minimum / positive outcome
    "neutral": "#6B4E71",     # muted plum — comparison / neutral category
}

# UI and legacy aliases used across the app
TERRACOTTA = PALETTE["terracotta"]
ACCENT = PALETTE["accent"]
DATA = PALETTE["data"]
MODEL = PALETTE["model"]
ERROR = PALETTE["error"]
GOOD = PALETTE["good"]
NEUTRAL = PALETTE["neutral"]
INK = PALETTE["ink"]
INK_SOFT = PALETTE["ink_soft"]

# Darkened Okabe–Ito categorical colours.  The hues remain colour-vision-
# deficiency aware and each swatch has at least 3:1 non-text contrast on both
# course backgrounds.  Pair colour with the marker/dash aliases below so
# meaning never relies on colour alone.
CB_BLUE = "#00689D"
CB_ORANGE = "#9C6800"
CB_GREEN = "#007A5A"
CB_SKY = "#397D9F"
CB_PURPLE = "#8C5278"
CB_VERMILLION = "#B64B00"
CB_GRAPHITE = "#3A3A3A"
CHART_SEQUENCE = (
    CB_BLUE,
    CB_VERMILLION,
    CB_GREEN,
    CB_PURPLE,
    CB_SKY,
    CB_ORANGE,
    CB_GRAPHITE,
)

# Semantic chart aliases.  Marker shapes and line styles are intentionally
# redundant with colour for students with colour-vision deficiencies.
TRAIN = CB_BLUE
VALIDATION = CB_VERMILLION
TEST = VALIDATION
CV = CB_GREEN
MODEL_A = PALETTE["accent"]
MODEL_B = PALETTE["neutral"]
BASELINE = PALETTE["error"]
TRAIN_MARKER = "circle"
VALIDATION_MARKER = "diamond"
TEST_MARKER = "diamond"
CV_MARKER = "square"
TRAIN_DASH = "solid"
VALIDATION_DASH = "dash"
TEST_DASH = "dash"
CV_DASH = "dot"
BASELINE_DASH = "dashdot"

# Shared continuous/fill colours used by diagnostics and feature views.
# Negative correlation is navy, zero blends into the page, and positive
# correlation uses the accessible terracotta accent.
CORRELATION_SCALE = (
    (0.0, DATA),
    (0.5, PALETTE["bg"]),
    (1.0, ACCENT),
)
GOOD_FADE = "rgba(62, 112, 99, 0.16)"

# Metric accent bars (match the course metric slide).
METRIC_ACCENTS = {
    "MAE": ACCENT,
    "MSE": ERROR,
    "RMSE": NEUTRAL,
    "R2": GOOD,
    "MAPE": DATA,
}


# ---------------------------------------------------------------------------
# Page-level CSS
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Apply the responsive cream/terracotta course theme."""
    p = PALETTE
    st.markdown(
        f"""
        <style>
        :root {{
            --rp-bg: {p['bg']};
            --rp-card: {p['card']};
            --rp-ink: {p['ink']};
            --rp-ink-soft: {p['ink_soft']};
            --rp-rule: {p['rule']};
            --rp-terracotta: {p['terracotta']};
            --rp-accent: {p['accent']};
        }}
        .stApp {{
            background: var(--rp-bg);
            color: var(--rp-ink);
            font-family: "Helvetica Neue", Arial, sans-serif;
        }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 1480px;
            padding-top: clamp(0.8rem, 2vw, 1.5rem);
            padding-right: clamp(1rem, 3vw, 3rem);
            padding-left: clamp(1rem, 3vw, 3rem);
        }}
        /* Headings retain the serif face used by the lecture slides. */
        h1, h2, h3, h4 {{
            font-family: Georgia, 'Times New Roman', serif !important;
            color: var(--rp-ink);
            line-height: 1.2;
        }}
        /* Compact app masthead: course context, product name, learning arc. */
        .rp-app-kicker {{
            color: var(--rp-accent);
            font-size: clamp(0.68rem, 1.1vw, 0.78rem);
            font-weight: 700;
            letter-spacing: 0.11em;
            line-height: 1.35;
            margin: 0.1rem 0 0.12rem;
            text-transform: uppercase;
        }}
        .rp-app-name {{
            color: var(--rp-ink);
            font-family: Georgia, 'Times New Roman', serif;
            font-size: clamp(1.85rem, 4vw, 2.75rem);
            font-weight: 700;
            letter-spacing: -0.025em;
            line-height: 1.08;
            margin: 0;
        }}
        .rp-app-tagline {{
            color: var(--rp-ink-soft);
            font-size: clamp(0.86rem, 1.5vw, 1rem);
            font-weight: 500;
            letter-spacing: 0.025em;
            line-height: 1.4;
            margin: 0.22rem 0 clamp(0.75rem, 1.8vw, 1.15rem);
        }}
        .rp-header {{
            padding: clamp(0.7rem, 1.8vw, 1rem) clamp(0.9rem, 2vw, 1.3rem);
            border-radius: 6px;
            border: 1px solid var(--rp-rule);
            border-left: 6px solid var(--rp-terracotta);
            background: var(--rp-card);
            margin-bottom: 0.75rem;
        }}
        .rp-title {{
            font-family: Georgia, 'Times New Roman', serif !important;
            font-size: clamp(1.45rem, 2.6vw, 1.85rem);
            font-weight: 700;
            letter-spacing: -0.015em;
            color: var(--rp-ink);
            margin: 0;
        }}
        .rp-sub {{
            color: var(--rp-ink-soft);
            font-size: 0.95rem;
            line-height: 1.45;
            margin: 0.2rem 0 0;
        }}
        /* Challenge / reveal call-out boxes */
        .rp-challenge {{
            background: #F6EFE4;
            border: 1px solid var(--rp-rule);
            border-left: 5px solid var(--rp-terracotta);
            padding: 0.65rem 0.9rem;
            border-radius: 5px;
            margin: 0.35rem 0 0.75rem;
            color: var(--rp-ink);
            line-height: 1.5;
        }}
        .rp-challenge-label {{
            color: var(--rp-accent);
            font-weight: 700;
        }}
        .rp-note {{
            color: var(--rp-ink-soft);
            font-size: 0.85rem;
            line-height: 1.45;
        }}
        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: var(--rp-card);
            border: 1px solid var(--rp-rule);
            border-top: 3px solid var(--rp-terracotta);
            padding: 0.65rem 0.8rem;
            border-radius: 5px;
            container: rp-metric / inline-size;
        }}
        div[data-testid="stMetric"] label p {{
            color: var(--rp-ink-soft) !important;
            font-weight: 600;
        }}
        div[data-testid="stMetricValue"] {{ color: var(--rp-ink); }}
        /*
         * The app uses metric deltas only for non-directional CV fold spread.
         * Streamlit otherwise inserts an up-arrow for a leading ± value, which
         * incorrectly suggests an improvement.
         */
        div[data-testid="stMetricDelta"] {{
            gap: 0 !important;
            padding-right: 0 !important;
            padding-left: 0 !important;
            color: var(--rp-ink-soft) !important;
        }}
        div[data-testid="stMetricDelta"] svg {{
            display: none !important;
        }}
        /*
         * Streamlit keeps metric labels/values on one line with ellipsis.
         * Some summaries sit beside charts and can become very narrow at
         * intermediate viewport widths. Adapt only those genuinely narrow
         * cards; full-width metric rows keep Streamlit's normal display size.
         */
        @container rp-metric (max-width: 8rem) {{
            label[data-testid="stMetricLabel"]
            [data-testid="stMarkdownContainer"],
            label[data-testid="stMetricLabel"] p {{
                overflow: visible !important;
                text-overflow: clip !important;
                white-space: normal !important;
                line-height: 1.15 !important;
            }}
            div[data-testid="stMetricValue"]
            [data-testid="stMarkdownContainer"],
            div[data-testid="stMetricValue"] p {{
                overflow: visible !important;
                text-overflow: clip !important;
                white-space: nowrap !important;
                font-size: 1.5rem !important;
                line-height: 1.25 !important;
            }}
            div[data-testid="stMetricDelta"]
            [data-testid="stMarkdownContainer"],
            div[data-testid="stMetricDelta"] p {{
                overflow: visible !important;
                text-overflow: clip !important;
                white-space: nowrap !important;
                font-size: 0.75rem !important;
                line-height: 1.2 !important;
            }}
        }}
        section[data-testid="stSidebar"] {{
            background: var(--rp-card);
            border-right: 1px solid var(--rp-rule);
        }}
        /* Workspace radio group presented as a vertical tab rail. */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            gap: 0.3rem;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"]
        label[data-baseweb="radio"] {{
            width: 100%;
            min-height: 2.45rem;
            padding: 0.52rem 0.65rem;
            border: 1px solid transparent;
            border-left: 4px solid transparent;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 120ms ease, border-color 120ms ease;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"]
        label[data-baseweb="radio"] > div:first-child {{
            display: none;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"]
        label[data-baseweb="radio"]:hover {{
            background: #F6EFE4;
            border-color: var(--rp-rule);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"]
        label[data-baseweb="radio"]:has(input:checked) {{
            background: #F6EFE4;
            border-color: var(--rp-rule);
            border-left-color: var(--rp-accent);
            color: var(--rp-ink);
            font-weight: 700;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"]
        label[data-baseweb="radio"]:has(input:focus-visible) {{
            outline: 3px solid var(--rp-accent);
            outline-offset: 2px;
        }}
        .stApp a {{
            color: var(--rp-accent);
            text-underline-offset: 0.15em;
        }}
        .stApp a:hover {{ color: #743914; }}
        .stApp a:focus-visible,
        .stApp button:focus-visible,
        .stApp input:focus-visible,
        .stApp textarea:focus-visible {{
            outline: 3px solid var(--rp-accent) !important;
            outline-offset: 2px;
        }}
        .rp-footer {{
            color: var(--rp-ink-soft);
            font-size: 0.8rem;
            border-top: 1px solid var(--rp-rule);
            padding-top: 0.5rem;
            margin-top: 1.2rem;
        }}
        @media (max-width: 700px) {{
            [data-testid="stMainBlockContainer"] {{
                /* Clear Streamlit's fixed mobile toolbar. */
                padding-top: 3.6rem;
                padding-right: 0.75rem;
                padding-left: 0.75rem;
            }}
            .rp-header {{
                border-left-width: 4px;
                margin-bottom: 0.6rem;
            }}
            .rp-app-kicker {{
                letter-spacing: 0.07em;
                margin-top: 0;
            }}
            .rp-app-name {{
                font-size: clamp(1.7rem, 9vw, 2.1rem);
            }}
            .rp-app-tagline {{
                margin-bottom: 0.7rem;
            }}
            .rp-challenge {{
                border-left-width: 4px;
                padding: 0.6rem 0.75rem;
            }}
            div[data-testid="stMetric"] {{
                padding: 0.55rem 0.65rem;
            }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str) -> None:
    """Render the visible workspace heading as a semantic level-one heading."""
    st.markdown(
        f"""
        <div class="rp-header">
            <h1 class="rp-title">{escape(title)}</h1>
            <p class="rp-sub">{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def challenge(text: str) -> None:
    """Render a challenge prompt."""
    st.markdown(
        (
            '<section class="rp-challenge" aria-label="Challenge">'
            '<span class="rp-challenge-label">Challenge&nbsp;·</span> '
            f"{escape(text)}</section>"
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------
def style_fig(fig, height: int = 360, title: str | None = None):
    """Apply the course look to a Plotly figure.

    The title text is always a string ("" when absent); passing ``None`` makes
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
        hoverlabel=dict(
            bgcolor=p["card"],
            bordercolor=p["rule"],
            font=dict(family="Helvetica Neue, Arial, sans-serif", color=p["ink"]),
        ),
        colorway=list(CHART_SEQUENCE),
    )
    fig.update_xaxes(gridcolor=p["grid"], zerolinecolor=p["rule"],
                     linecolor=p["rule"], tickcolor=p["rule"])
    fig.update_yaxes(gridcolor=p["grid"], zerolinecolor=p["rule"],
                     linecolor=p["rule"], tickcolor=p["rule"])
    return fig

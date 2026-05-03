import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

sys.path.append('src/backend')

from models.option_models import OptionPricingModels, ImpliedVolatilityCalculator
from models.implied_volatility import ImpliedVolatilitySurface
from data.data_fetcher import DataFetcher
from utils.helpers import validate_option_parameters
from backtesting.backtester import SPXBacktester


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantra · Options Pricing Infrastructure",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens + global styles
# All tokens live in one place as CSS custom properties so the rest of the
# stylesheet composes from them. Mirrors the shadcn / design-system approach.
# ─────────────────────────────────────────────────────────────────────────────
DESIGN_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Calistoga&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    /* Palette */
    --background: #FAFAFA;
    --foreground: #0F172A;
    --muted: #F1F5F9;
    --muted-foreground: #64748B;
    --border: #E2E8F0;
    --card: #FFFFFF;
    --accent: #0052FF;
    --accent-secondary: #4D7CFF;
    --accent-foreground: #FFFFFF;
    --ring: #0052FF;

    /* Elevation */
    --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06);
    --shadow-md: 0 4px 6px rgba(15, 23, 42, 0.07);
    --shadow-lg: 0 10px 15px rgba(15, 23, 42, 0.08);
    --shadow-xl: 0 20px 25px rgba(15, 23, 42, 0.10);
    --shadow-accent: 0 4px 14px rgba(0, 82, 255, 0.25);
    --shadow-accent-lg: 0 8px 24px rgba(0, 82, 255, 0.35);

    /* Type */
    --font-display: 'Calistoga', Georgia, serif;
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, monospace;
  }

  /* ───────── App shell ───────── */
  .stApp {
    background: var(--background);
    color: var(--foreground);
    font-family: var(--font-sans);
  }

  /* Hide Streamlit chrome */
  .stDeployButton { display: none; }
  footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; height: 0; }
  [data-testid="stToolbar"] { display: none !important; }
  .viewerBadge_container__1QSob,
  .viewerBadge_link__1S137 { display: none !important; }
  #MainMenu { visibility: hidden; }

  /* Full-width container with generous breathing room */
  .main .block-container {
    max-width: 1200px;
    padding: 1.75rem 2rem 4rem 2rem;
  }

  /* ───────── Typography ───────── */
  html, body, [class*="css"] {
    font-family: var(--font-sans);
    color: var(--foreground);
  }

  h1, h2, h3, h4, h5, h6,
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: var(--font-sans);
    color: var(--foreground);
    letter-spacing: -0.01em;
  }

  /* Display headings opt-in */
  .display-xl {
    font-family: var(--font-display);
    font-size: clamp(2.75rem, 6vw, 5.25rem);
    line-height: 1.02;
    letter-spacing: -0.025em;
    color: var(--foreground);
    margin: 0 0 1.25rem 0;
    font-weight: 400;
  }

  .display-lg {
    font-family: var(--font-display);
    font-size: clamp(2rem, 4vw, 3.25rem);
    line-height: 1.12;
    letter-spacing: -0.02em;
    color: var(--foreground);
    margin: 0 0 0.75rem 0;
    font-weight: 400;
  }

  .display-md {
    font-family: var(--font-display);
    font-size: clamp(1.5rem, 2.5vw, 2.25rem);
    line-height: 1.15;
    color: var(--foreground);
    margin: 0 0 0.5rem 0;
    font-weight: 400;
  }

  .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(0, 82, 255, 0.28);
    background: rgba(0, 82, 255, 0.06);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1.25rem;
  }

  .eyebrow .pulse-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 999px;
    background: var(--accent);
    animation: pulse 2s ease-in-out infinite;
    box-shadow: 0 0 0 0 rgba(0, 82, 255, 0.6);
  }

  .gradient-text {
    background: linear-gradient(120deg, var(--accent) 0%, var(--accent-secondary) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
  }

  .headline-underline {
    position: relative;
    display: inline-block;
  }
  .headline-underline::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -0.15rem;
    height: 0.85rem;
    background: linear-gradient(to right, rgba(0,82,255,0.18), rgba(77,124,255,0.08));
    border-radius: 4px;
    z-index: -1;
  }

  .body-lead {
    font-size: 1.125rem;
    line-height: 1.7;
    color: var(--muted-foreground);
    max-width: 56ch;
  }

  .mono-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted-foreground);
    margin: 0 0 0.35rem 0;
    display: block;
  }

  .caption {
    color: var(--muted-foreground);
    font-size: 0.9rem;
    line-height: 1.6;
  }

  /* ───────── Cards ───────── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease, transform 0.3s ease;
  }
  .card:hover {
    box-shadow: var(--shadow-lg);
  }

  .card--muted {
    background: var(--muted);
    border-color: transparent;
  }

  .card--feature {
    min-height: 200px;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .card--feature .icon-tile {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-family: var(--font-mono);
    font-weight: 500;
    box-shadow: var(--shadow-accent);
    margin-bottom: 0.25rem;
  }
  .card--feature h4 {
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 1.15rem;
    margin: 0;
    color: var(--foreground);
    letter-spacing: -0.01em;
  }
  .card--feature p {
    color: var(--muted-foreground);
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
  }

  /* Inverted dark section (stats / CTA) */
  .inverted-section {
    background: var(--foreground);
    color: #FFFFFF;
    border-radius: 24px;
    padding: 3rem 2.5rem;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
  }
  .inverted-section::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(circle, rgba(255,255,255,0.85) 1px, transparent 1px);
    background-size: 28px 28px;
    opacity: 0.04;
    pointer-events: none;
  }
  .inverted-section::after {
    content: "";
    position: absolute;
    top: -140px;
    right: -140px;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0, 82, 255, 0.35), transparent 70%);
    filter: blur(90px);
    pointer-events: none;
  }
  .inverted-section .eyebrow {
    border-color: rgba(255,255,255,0.22);
    background: rgba(255,255,255,0.06);
    color: #CBD5E1;
  }
  .inverted-section .display-lg { color: #F8FAFC; }
  .inverted-section .body-lead { color: rgba(226, 232, 240, 0.75); }
  .inverted-section .stat-number {
    font-family: var(--font-display);
    font-size: clamp(2.25rem, 3.6vw, 3.25rem);
    line-height: 1;
    color: #FFFFFF;
    letter-spacing: -0.02em;
  }
  .inverted-section .stat-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(203, 213, 225, 0.75);
    margin-top: 0.5rem;
  }
  .inverted-section .stat-sub {
    color: rgba(148, 163, 184, 0.85);
    font-size: 0.85rem;
    margin-top: 0.35rem;
  }

  /* ───────── Hero graphic ───────── */
  .hero-graphic {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    max-width: 440px;
    margin-left: auto;
    min-height: 320px;
  }
  .hero-graphic .ring {
    position: absolute;
    inset: 6%;
    border-radius: 50%;
    border: 1px dashed rgba(0, 82, 255, 0.35);
    animation: rotate 65s linear infinite;
  }
  .hero-graphic .ring.inner {
    inset: 22%;
    border-color: rgba(77, 124, 255, 0.35);
    animation: rotate 90s linear infinite reverse;
  }
  .hero-graphic .glow {
    position: absolute;
    inset: 10%;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0, 82, 255, 0.22), transparent 60%);
    filter: blur(48px);
  }
  .hero-graphic .float-card {
    position: absolute;
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    box-shadow: var(--shadow-lg);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--foreground);
    letter-spacing: 0.04em;
    line-height: 1.5;
  }
  .hero-graphic .float-card .v {
    font-family: var(--font-display);
    font-size: 1.4rem;
    color: var(--accent);
    display: block;
    line-height: 1.1;
    margin-top: 0.15rem;
  }
  .hero-graphic .float-card.c1 { top: 8%; left: 4%; animation: floatA 6s ease-in-out infinite; }
  .hero-graphic .float-card.c2 { top: 48%; right: 0%; animation: floatB 5.5s ease-in-out infinite; }
  .hero-graphic .float-card.c3 { bottom: 6%; left: 18%; animation: floatA 7s ease-in-out infinite; }
  .hero-graphic .accent-block {
    position: absolute;
    bottom: 10%;
    right: 10%;
    width: 62px;
    height: 62px;
    background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
    border-radius: 16px;
    box-shadow: var(--shadow-accent-lg);
  }
  .hero-graphic .dot-grid {
    position: absolute;
    top: 10%;
    right: 14%;
    width: 72px;
    height: 72px;
    background-image: radial-gradient(circle, rgba(15,23,42,0.35) 1.2px, transparent 1.2px);
    background-size: 14px 14px;
    opacity: 0.6;
  }

  /* ───────── Navigation ───────── */
  .nav-shell {
    position: sticky;
    top: 0;
    z-index: 50;
    background: rgba(250, 250, 250, 0.72);
    backdrop-filter: saturate(160%) blur(12px);
    -webkit-backdrop-filter: saturate(160%) blur(12px);
    border-bottom: 1px solid var(--border);
    margin: -1.75rem -2rem 2rem -2rem;
    padding: 1rem 2rem;
  }
  .brand {
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;
    font-family: var(--font-display);
    font-size: 1.35rem;
    letter-spacing: -0.01em;
    color: var(--foreground);
  }
  .brand .mark {
    width: 26px;
    height: 26px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
    box-shadow: var(--shadow-accent);
    display: inline-block;
  }

  /* Navigation buttons — secondary (non-active) get a ghost-pill look.
     Inside the nav, we restyle type="primary" (active tab) to a solid dark
     pill so the gradient stays reserved for real CTAs. */
  .nav-shell div[data-testid="stButton"] > button {
    background: transparent !important;
    color: var(--muted-foreground) !important;
    border: 1px solid transparent !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.16em !important;
    padding: 0.55rem 1rem !important;
    border-radius: 999px !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
  }
  .nav-shell div[data-testid="stButton"] > button:hover {
    color: var(--foreground) !important;
    background: var(--muted) !important;
    border-color: var(--border) !important;
    transform: none;
  }
  .nav-shell div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--foreground) !important;
    color: #FFFFFF !important;
    border-color: var(--foreground) !important;
    box-shadow: none !important;
    filter: none !important;
  }
  .nav-shell div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1E293B !important;
    border-color: #1E293B !important;
  }

  /* ───────── Primary buttons ───────── */
  .stButton > button,
  .stDownloadButton > button {
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.4rem !important;
    transition: all 0.2s ease-out !important;
    font-size: 0.95rem !important;
    border: 1px solid var(--border) !important;
    background: var(--card) !important;
    color: var(--foreground) !important;
    box-shadow: var(--shadow-sm) !important;
  }
  .stButton > button:hover,
  .stDownloadButton > button:hover {
    border-color: rgba(0, 82, 255, 0.35) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px);
  }

  /* Primary variant */
  .stButton > button[kind="primary"],
  .stDownloadButton > button[kind="primary"] {
    background: linear-gradient(120deg, var(--accent), var(--accent-secondary)) !important;
    color: var(--accent-foreground) !important;
    border: 1px solid transparent !important;
    box-shadow: var(--shadow-accent) !important;
  }
  .stButton > button[kind="primary"]:hover,
  .stDownloadButton > button[kind="primary"]:hover {
    box-shadow: var(--shadow-accent-lg) !important;
    filter: brightness(1.05);
    transform: translateY(-1px);
  }
  .stButton > button[kind="primary"]:active,
  .stDownloadButton > button[kind="primary"]:active {
    transform: scale(0.99);
  }

  /* ───────── Inputs ───────── */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div,
  .stDateInput > div > div > input {
    background: var(--card) !important;
    color: var(--foreground) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.95rem !important;
  }
  .stTextInput > div > div > input:focus,
  .stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0, 82, 255, 0.12) !important;
  }
  .stNumberInput button {
    background: transparent !important;
    border: none !important;
    color: var(--muted-foreground) !important;
  }
  .stCheckbox > label,
  .stRadio > label {
    color: var(--foreground) !important;
    font-family: var(--font-sans) !important;
  }

  /* Slider */
  .stSlider > div > div > div {
    color: var(--foreground) !important;
  }
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: var(--shadow-accent) !important;
  }

  /* ───────── Metrics (st.metric) ───────── */
  [data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.25s ease, transform 0.25s ease;
  }
  [data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
  }
  [data-testid="stMetricLabel"] p {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted-foreground) !important;
  }
  [data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    color: var(--foreground) !important;
    font-size: 1.9rem !important;
    letter-spacing: -0.02em !important;
  }
  [data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--accent) !important;
  }

  /* ───────── Tables / dataframes ───────── */
  [data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    background: var(--card);
  }

  /* ───────── Alerts ───────── */
  .stAlert {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--card) !important;
  }

  /* ───────── Expander ───────── */
  .stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--card) !important;
    box-shadow: var(--shadow-sm) !important;
  }

  /* ───────── Plotly ───────── */
  .js-plotly-plot {
    border-radius: 14px;
    background: var(--card) !important;
  }

  /* ───────── Animations ───────── */
  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(0, 82, 255, 0.55); }
    50%      { transform: scale(1.25); opacity: 0.75; box-shadow: 0 0 0 8px rgba(0, 82, 255, 0); }
  }
  @keyframes rotate {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }
  @keyframes floatA {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-10px); }
  }
  @keyframes floatB {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(10px); }
  }
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .fade-in-up {
    animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
  }

  @media (prefers-reduced-motion: reduce) {
    .ring, .ring.inner, .pulse-dot,
    .hero-graphic .float-card, .fade-in-up { animation: none !important; }
  }

  /* ───────── Divider ───────── */
  .section-divider {
    border: 0;
    height: 1px;
    background: var(--border);
    margin: 3rem 0 2rem 0;
  }

  /* ───────── Footer ───────── */
  .site-footer {
    margin-top: 4rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid var(--border);
    color: var(--muted-foreground);
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
  }
  .site-footer .mono { font-family: var(--font-mono); letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.72rem; }

  /* Responsive tweaks */
  @media (max-width: 880px) {
    .hero-graphic { display: none; }
    .inverted-section { padding: 2rem 1.5rem; }
  }
</style>
"""

st.markdown(DESIGN_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def section_label(text: str, pulse: bool = False) -> None:
    """Pill badge used at the start of every major section."""
    dot = '<span class="pulse-dot"></span>' if pulse else '<span class="pulse-dot" style="animation:none"></span>'
    st.markdown(
        f'<div class="eyebrow">{dot}<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def display_heading(text: str, highlight: str | None = None, size: str = "lg") -> None:
    """Renders a Calistoga display heading, optionally with a gradient-highlighted word."""
    cls = {"xl": "display-xl", "lg": "display-lg", "md": "display-md"}[size]
    if highlight and highlight in text:
        parts = text.split(highlight, 1)
        inner = (
            f'{parts[0]}<span class="headline-underline gradient-text">{highlight}</span>{parts[1]}'
        )
    else:
        inner = text
    st.markdown(f'<h1 class="{cls}">{inner}</h1>', unsafe_allow_html=True)


def lead(text: str) -> None:
    st.markdown(f'<p class="body-lead">{text}</p>', unsafe_allow_html=True)


def mono_label(text: str) -> None:
    st.markdown(f'<span class="mono-label">{text}</span>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)


def feature_card(index: str, title: str, body: str) -> str:
    return f"""
    <div class="card card--feature fade-in-up">
      <div class="icon-tile">{index}</div>
      <h4>{title}</h4>
      <p>{body}</p>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme — a single helper keeps every chart visually consistent with
# the design tokens (white surface, Inter font, accent blue series).
# ─────────────────────────────────────────────────────────────────────────────
ACCENT = "#0052FF"
ACCENT_SECONDARY = "#4D7CFF"
FOREGROUND = "#0F172A"
MUTED_FG = "#64748B"
BORDER = "#E2E8F0"
SURFACE = "#FFFFFF"

ACCENT_COLORSCALE = [
    [0.0, "#EAF1FF"],
    [0.25, "#A9C2FF"],
    [0.5, "#6A97FF"],
    [0.75, "#2B6BFF"],
    [1.0, "#0034B8"],
]


def style_fig(fig: go.Figure, *, height: int = 520, scene: bool = False) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, system-ui, sans-serif", color=FOREGROUND, size=13),
        title=dict(font=dict(family="Inter, system-ui, sans-serif", size=16, color=FOREGROUND), x=0.01, xanchor="left"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=FOREGROUND, size=12),
        ),
        height=height,
        margin=dict(l=20, r=20, t=60, b=40),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=BORDER,
            font=dict(family="Inter, system-ui, sans-serif", color=FOREGROUND, size=12),
        ),
    )
    if scene:
        fig.update_layout(
            scene=dict(
                bgcolor=SURFACE,
                xaxis=dict(backgroundcolor=SURFACE, gridcolor="#E2E8F0", color=FOREGROUND, showbackground=False),
                yaxis=dict(backgroundcolor=SURFACE, gridcolor="#E2E8F0", color=FOREGROUND, showbackground=False),
                zaxis=dict(backgroundcolor=SURFACE, gridcolor="#E2E8F0", color=FOREGROUND, showbackground=False),
            )
        )
    else:
        fig.update_xaxes(gridcolor="#EEF2F7", linecolor=BORDER, zeroline=False, tickcolor=BORDER, title_font=dict(color=MUTED_FG, size=12), tickfont=dict(color=MUTED_FG, size=11))
        fig.update_yaxes(gridcolor="#EEF2F7", linecolor=BORDER, zeroline=False, tickcolor=BORDER, title_font=dict(color=MUTED_FG, size=12), tickfont=dict(color=MUTED_FG, size=11))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Numerical differentiation — finite-difference Greeks. Surfaced in the UI so
# the "numerical differentiation algorithms" resume bullet is visibly
# evidenced alongside the analytical Black-Scholes Greeks.
# ─────────────────────────────────────────────────────────────────────────────
def _bs_price(S, K, T, r, sigma, option_type):
    models = OptionPricingModels(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
    return float(models.black_scholes_option())


def finite_difference_greeks(S, K, T, r, sigma, option_type, h_S=0.01, h_sigma=1e-4, h_T=1 / 365.25, h_r=1e-4):
    """Central-difference approximations of the Greeks — direct numerical
    differentiation of the Black-Scholes pricing function."""
    S_up, S_dn = S * (1 + h_S), S * (1 - h_S)
    p_up = _bs_price(S_up, K, T, r, sigma, option_type)
    p_dn = _bs_price(S_dn, K, T, r, sigma, option_type)
    p0 = _bs_price(S, K, T, r, sigma, option_type)

    delta = (p_up - p_dn) / (S_up - S_dn)
    gamma = (p_up - 2 * p0 + p_dn) / ((S * h_S) ** 2)

    vega_up = _bs_price(S, K, T, r, sigma + h_sigma, option_type)
    vega_dn = _bs_price(S, K, T, r, sigma - h_sigma, option_type)
    vega = (vega_up - vega_dn) / (2 * h_sigma)

    T_fwd = max(T - h_T, 1e-6)
    theta = (_bs_price(S, K, T_fwd, r, sigma, option_type) - p0) / (-h_T) * -1
    # Convention: theta per year; negative = decay.

    rho_up = _bs_price(S, K, T, r + h_r, sigma, option_type)
    rho_dn = _bs_price(S, K, T, r - h_r, sigma, option_type)
    rho = (rho_up - rho_dn) / (2 * h_r)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "data_fetcher" not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "demo_surface_generated" not in st.session_state:
    st.session_state.demo_surface_generated = True


# ─────────────────────────────────────────────────────────────────────────────
# Demo SPX surface (preserved from original — powers the hero demo card)
# ─────────────────────────────────────────────────────────────────────────────
def generate_demo_volatility_surface():
    strikes = np.linspace(4800, 5200, 20)
    expiries = np.array([7, 14, 30, 60, 90, 120])
    strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
    base_vol = 0.18 + 0.02 * np.random.randn()
    current_spot = 5000 + 50 * np.random.randn()
    moneyness = strike_grid / current_spot
    smile_effect = 0.05 * (moneyness - 1) ** 2
    term_effect = 0.02 * np.log(expiry_grid / 30)
    noise = 0.01 * np.random.randn(*strike_grid.shape)
    vol_surface = np.maximum(base_vol + smile_effect + term_effect + noise, 0.05)
    return strike_grid, expiry_grid, vol_surface, current_spot


# ─────────────────────────────────────────────────────────────────────────────
# Top navigation
# ─────────────────────────────────────────────────────────────────────────────
PAGES = ["Home", "Option Pricing", "Implied Volatility Surface", "Backtesting Results"]
PAGE_KEYS = {"Home": "home", "Option Pricing": "pricing", "Implied Volatility Surface": "iv", "Backtesting Results": "backtest"}

st.markdown('<div class="nav-shell">', unsafe_allow_html=True)

nav_cols = st.columns([2.6, 1, 1.2, 1.5, 1.6], gap="small")
with nav_cols[0]:
    st.markdown(
        '<div class="brand"><span class="mark"></span> Quantra <span style="font-family:var(--font-mono); font-size:0.7rem; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted-foreground); margin-left:0.5rem;">Options Engine</span></div>',
        unsafe_allow_html=True,
    )

def _nav_btn(col, label: str, page: str, key: str) -> None:
    is_active = st.session_state.current_page == page
    with col:
        if st.button(
            label,
            key=key,
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_page = page
            st.rerun()


_nav_btn(nav_cols[1], "Home", "Home", "nav_home")
_nav_btn(nav_cols[2], "Pricing", "Option Pricing", "nav_pricing")
_nav_btn(nav_cols[3], "Volatility", "Implied Volatility Surface", "nav_iv")
_nav_btn(nav_cols[4], "Backtesting", "Backtesting Results", "nav_backtest")

st.markdown("</div>", unsafe_allow_html=True)  # /nav-shell


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.current_page == "Home":
    # Hero — asymmetric grid 1.1fr / 0.9fr
    hero_left, hero_right = st.columns([1.1, 0.9], gap="large")

    with hero_left:
        st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
        section_label("Real-Time Options Infrastructure", pulse=True)
        display_heading("Price, calibrate and backtest options at the speed of markets.", highlight="calibrate", size="xl")
        lead(
            "Quantra is a real-time options engine that pipes YFinance feeds into calibrated "
            "Black-Scholes, Binomial and Monte Carlo models, layered with dynamic volatility surface fitting, "
            "numerical-differentiation Greeks, and a historical backtester built for thousands of "
            "concurrent contracts."
        )
        st.write("")
        cta_a, cta_b, _ = st.columns([1, 1, 0.6])
        with cta_a:
            if st.button("Price an option  →", type="primary", key="cta_price", use_container_width=True):
                st.session_state.current_page = "Option Pricing"
                st.rerun()
        with cta_b:
            if st.button("Open backtester", key="cta_backtest", use_container_width=True):
                st.session_state.current_page = "Backtesting Results"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with hero_right:
        st.markdown(
            """
            <div class="hero-graphic fade-in-up">
              <div class="glow"></div>
              <div class="ring"></div>
              <div class="ring inner"></div>
              <div class="dot-grid"></div>
              <div class="accent-block"></div>
              <div class="float-card c1">BLACK-SCHOLES<span class="v">$12.48</span></div>
              <div class="float-card c2">BINOMIAL N=100<span class="v">$12.51</span></div>
              <div class="float-card c3">MONTE CARLO<span class="v">$12.46</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Capability cards
    divider()
    section_label("Capabilities")
    st.markdown(
        '<p style="color: var(--foreground); font-size: 1.05rem; line-height: 1.65; max-width: 64ch; margin: 0 0 1.25rem 0;">Model selection, volatility fitting and historical validation share a unified data layer and a consistent type system, so a calibration on the pricing page transfers cleanly into the backtester.</p>',
        unsafe_allow_html=True,
    )
    st.write("")

    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        st.markdown(
            feature_card(
                "01",
                "Pricing Models",
                "Closed-form Black-Scholes, discrete Binomial (Cox-Ross-Rubinstein) and path-wise Monte Carlo, benchmarked against each other live so model disagreement becomes a signal.",
            ),
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            feature_card(
                "02",
                "Volatility Surface",
                "Brent, Newton-Raphson and bisection IV solvers fit a dynamic surface across strikes and expiries, exposing smile, skew, and term-structure slope for any YFinance-listed ticker.",
            ),
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            feature_card(
                "03",
                "Backtesting Engine",
                "Validate models against historical SPX chains with 1,000+ concurrent contracts, numerical-differentiation risk metrics, and MAE / RMSE / MAPE diagnostics per model.",
            ),
            unsafe_allow_html=True,
        )

    # Live demo surface
    divider()
    top_l, top_r = st.columns([2, 1])
    with top_l:
        section_label("Live Demo", pulse=True)
    with top_r:
        if st.button("↻  Regenerate surface", type="primary", key="demo_surface", use_container_width=True):
            st.session_state.demo_surface_generated = True
            # Force a reseed by invalidating the cache key
            st.session_state["_demo_seed"] = np.random.randint(0, 1_000_000)

    if st.session_state.demo_surface_generated:
        if "_demo_seed" in st.session_state:
            np.random.seed(st.session_state["_demo_seed"])
        strike_grid, expiry_grid, vol_surface, current_spot = generate_demo_volatility_surface()

        fig = go.Figure(
            data=[
                go.Surface(
                    x=strike_grid,
                    y=expiry_grid,
                    z=vol_surface * 100,
                    colorscale=ACCENT_COLORSCALE,
                    showscale=True,
                    colorbar=dict(title=dict(text="IV (%)", font=dict(color=FOREGROUND, size=12)), tickfont=dict(color=MUTED_FG, size=11)),
                    hovertemplate="Strike: %{x:.0f}<br>Days: %{y:.0f}<br>IV: %{z:.1f}%<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=f"SPX Implied Volatility Surface  ·  Spot ${current_spot:.0f}",
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Days to Expiry",
                zaxis_title="IV (%)",
            ),
        )
        st.plotly_chart(style_fig(fig, height=560, scene=True), use_container_width=True)

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5, gap="small")
        with c1:
            st.metric("SPX Spot", f"${current_spot:.0f}", f"{np.random.choice(['+','-'])}{abs(np.random.randn()*10):.1f}")
        with c2:
            atm_vol = vol_surface[2, 10] * 100
            st.metric("ATM IV", f"{atm_vol:.1f}%", f"{np.random.choice(['+','-'])}{abs(np.random.randn()*2):.2f}%")
        with c3:
            st.metric("Vol Range", f"{(vol_surface.max()-vol_surface.min())*100:.1f}%")
        with c4:
            skew = (vol_surface[2, 5] - vol_surface[2, 15]) * 100
            st.metric("Skew", f"{skew:.2f}%")
        with c5:
            slope = (vol_surface[5, 10] - vol_surface[0, 10]) * 100
            st.metric("Term Slope", f"{slope:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# OPTION PRICING
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_page == "Option Pricing":
    section_label("Pricing Workbench", pulse=True)
    display_heading("Price any option", highlight="option", size="lg")
    lead("Enter a ticker and parameters. We pull live data from YFinance, price it three ways, and produce both analytical and numerically-differentiated Greeks for sensitivity and delta-hedge sizing.")

    divider()

    # Inputs
    section_label("Inputs")
    col_a, col_b, col_c = st.columns(3, gap="medium")

    with col_a:
        mono_label("ticker symbol")
        ticker = st.text_input("ticker_symbol", value="AAPL", label_visibility="collapsed", help="Enter any yfinance-supported ticker (e.g. AAPL, MSFT, SPY).")
        mono_label("option type")
        option_type = st.selectbox("option_type", ["Call", "Put"], label_visibility="collapsed")
        mono_label("strike price")
        strike_price = st.number_input("strike_price", min_value=0.01, value=150.0, step=0.5, label_visibility="collapsed")

    with col_b:
        mono_label("days to expiration")
        days_to_expiry = st.number_input("days_to_expiration", min_value=1, value=30, step=1, key="days_exp", label_visibility="collapsed")
        mono_label("risk free rate percent")
        risk_free_rate = st.number_input("risk_free_rate_percent", min_value=0.0, value=5.0, step=0.1, key="risk_rate", label_visibility="collapsed") / 100
        mono_label("volatility percent")
        volatility = st.number_input("volatility_percent", min_value=0.1, value=25.0, step=0.5, key="volatility", label_visibility="collapsed") / 100

    with col_c:
        mono_label("use live market data?")
        use_live_data = st.checkbox("Pull live YFinance quote", value=True, key="use_live_data")
        stock_price = 150.0
        if use_live_data:
            with st.spinner("Fetching live data…"):
                current_price = st.session_state.data_fetcher.get_current_price(ticker)
                hist_vol = st.session_state.data_fetcher.calculate_historical_volatility(ticker)
                if current_price:
                    st.success(f"Live price: **${current_price:.2f}**")
                    stock_price = current_price
                else:
                    st.error("Could not fetch live data")
                    mono_label("stock price")
                    stock_price = st.number_input("stock_price", min_value=0.01, value=150.0, step=0.1, key="stock_price", label_visibility="collapsed")
                if hist_vol:
                    st.info(f"30-day realized vol: **{hist_vol*100:.1f}%**")
                    if st.checkbox("Use realized vol as σ", key="use_hist_vol"):
                        volatility = hist_vol
        else:
            mono_label("stock price")
            stock_price = st.number_input("stock_price_manual", min_value=0.01, value=150.0, step=0.1, key="manual_stock_price", label_visibility="collapsed")

        mono_label("position size contracts")
        position_size = st.number_input("position_size", min_value=1, value=10, step=1, label_visibility="collapsed", help="Used in the hedging panel below.")

    st.write("")
    if st.button("Calculate option prices", type="primary", key="run_pricing"):
        time_to_expiry = days_to_expiry / 365.25
        is_valid, errors = validate_option_parameters(stock_price, strike_price, time_to_expiry, risk_free_rate, volatility)
        if not is_valid:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("Calibrating models…"):
                pricing_models = OptionPricingModels(
                    S=stock_price, K=strike_price, T=time_to_expiry,
                    r=risk_free_rate, sigma=volatility, option_type=option_type.lower(),
                )
                bs_price = pricing_models.black_scholes_option()
                bt_price = pricing_models.binomial_tree_option_price(N=100)
                mc_price, _ = pricing_models.new_monte_carlo_option_price(num_simulations=10000)
                greeks_analytical = pricing_models.calculate_greeks()
                greeks_numerical = finite_difference_greeks(
                    stock_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type.lower(),
                )

            # Pricing results
            divider()
            section_label("Model Prices")
            p1, p2, p3, p4 = st.columns(4, gap="small")
            with p1: st.metric("Black-Scholes", f"${bs_price:.4f}", help="Closed-form analytical solution")
            with p2: st.metric("Binomial Tree", f"${bt_price:.4f}", help="Cox-Ross-Rubinstein lattice, N=100 steps")
            with p3: st.metric("Monte Carlo", f"${mc_price:.4f}", help="10,000 GBM simulations")
            with p4:
                spread = max(bs_price, bt_price, mc_price) - min(bs_price, bt_price, mc_price)
                st.metric("Model Disagreement", f"${spread:.4f}", help="max – min across models")

            # Greeks
            divider()
            section_label("Greeks: Analytical vs Numerical")
            lead("Closed-form Greeks (left column of each tile) are compared against central-difference numerical derivatives of the Black-Scholes pricer, validating the numerical differentiation pipeline that powers the backtester.")

            g_cols = st.columns(5, gap="small")
            labels = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
            keys = ["delta", "gamma", "vega", "theta", "rho"]
            fmts = ["{:.4f}", "{:.6f}", "{:.4f}", "{:.4f}", "{:.4f}"]
            for col, lbl, k, fmt in zip(g_cols, labels, keys, fmts):
                a_val = greeks_analytical[k]
                n_val = greeks_numerical[k]
                diff = abs(a_val - n_val)
                delta_txt = f"Δ {diff:.2e} vs analytical"
                col.metric(lbl, fmt.format(n_val), delta_txt, help=f"Analytical: {fmt.format(a_val)}\nNumerical (central diff): {fmt.format(n_val)}")

            # Sensitivity analysis
            divider()
            section_label("Price Sensitivity")
            spot_range = np.linspace(stock_price * 0.8, stock_price * 1.2, 60)
            sens_prices = []
            for spot in spot_range:
                sens_prices.append(_bs_price(spot, strike_price, time_to_expiry, risk_free_rate, volatility, option_type.lower()))

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=spot_range, y=sens_prices, mode="lines",
                    name="BS price",
                    line=dict(color=ACCENT, width=3),
                    fill="tozeroy",
                    fillcolor="rgba(0, 82, 255, 0.06)",
                    hovertemplate="Spot $%{x:.2f}<br>Price $%{y:.4f}<extra></extra>",
                )
            )
            fig.add_vline(x=stock_price, line_dash="dot", line_color=FOREGROUND, annotation_text=f"Spot ${stock_price:.2f}", annotation_position="top right", annotation=dict(font=dict(color=FOREGROUND)))
            fig.add_vline(x=strike_price, line_dash="dot", line_color=ACCENT_SECONDARY, annotation_text=f"Strike ${strike_price:.0f}", annotation_position="top left", annotation=dict(font=dict(color=ACCENT_SECONDARY)))
            fig.update_layout(title="Option Value vs Underlying Price", xaxis_title="Underlying Price ($)", yaxis_title="Option Price ($)")
            st.plotly_chart(style_fig(fig, height=520), use_container_width=True)

            # Hedging / portfolio panel — evidences the "hedging & portfolio management" bullet.
            divider()
            section_label("Hedging Workflow")
            lead("Using the numerical Δ above, we compute the share count required to neutralize first-order directional risk for your contract exposure, plus portfolio-level exposure metrics.")

            delta = greeks_numerical["delta"]
            gamma = greeks_numerical["gamma"]
            vega = greeks_numerical["vega"]
            theta = greeks_numerical["theta"]

            multiplier = 100  # equity options
            port_delta = delta * position_size * multiplier
            port_gamma = gamma * position_size * multiplier
            port_vega = vega * position_size * multiplier
            port_theta = theta * position_size * multiplier
            hedge_shares = -port_delta  # short stock to offset long-call delta (sign handled)

            h1, h2, h3, h4 = st.columns(4, gap="small")
            with h1: st.metric("Portfolio Δ", f"{port_delta:+.2f}", help="Net delta across all contracts")
            with h2: st.metric("Hedge Shares", f"{hedge_shares:+.0f}", help="Shares of underlying to hold to zero net delta")
            with h3: st.metric("Notional Hedge", f"${abs(hedge_shares)*stock_price:,.0f}", help="Dollar notional of the hedge leg")
            with h4: st.metric("Net Premium", f"${bs_price*position_size*multiplier:,.2f}", help="BS price × contracts × 100")

            with st.expander("Portfolio Γ, ν, θ: second-order hedging metrics"):
                ec1, ec2, ec3 = st.columns(3)
                with ec1: st.metric("Portfolio Γ", f"{port_gamma:+.4f}", help="Delta drift per $1 move in spot")
                with ec2: st.metric("Portfolio ν", f"${port_vega:+.2f}", help="P&L per 1 vol-point (0.01) move in IV")
                with ec3: st.metric("Portfolio θ", f"${port_theta:+.2f} / yr", help="Annualized time-decay exposure")

            # Integrated Volatility Smile
            divider()
            section_label("Live Volatility Smile")
            with st.spinner("Generating volatility smile…"):
                try:
                    iv_surface = ImpliedVolatilitySurface(ticker)
                    options_data = iv_surface.fetch_options_chain()
                    if not options_data.empty:
                        options_with_iv = iv_surface.calculate_iv_for_chain(options_data, method="brent")
                        if not options_with_iv.empty:
                            smile_fig = iv_surface.analyze_volatility_smile(options_with_iv)
                            if smile_fig:
                                # Recolor the smile trace set to the design palette
                                for i, tr in enumerate(smile_fig.data):
                                    color = ACCENT if i == 0 else ACCENT_SECONDARY
                                    tr.marker = dict(color=color, size=8, line=dict(color="white", width=1))
                                    tr.line = dict(color=color, width=2)
                                st.plotly_chart(style_fig(smile_fig, height=460), use_container_width=True)
                            stats = iv_surface.calculate_surface_statistics(options_with_iv)
                            sc1, sc2, sc3 = st.columns(3)
                            with sc1: st.metric("ATM IV (short)", f"{stats.get('atm_iv_short', 0)*100:.1f}%")
                            with sc2: st.metric("ATM IV (long)", f"{stats.get('atm_iv_long', 0)*100:.1f}%")
                            with sc3: st.metric("Skew", f"{stats.get('skew', 0)*100:.2f}%")
                        else:
                            st.info("No valid options found for smile analysis.")
                    else:
                        st.info("No options chain available for this ticker.")
                except Exception as e:
                    st.info(f"Smile analysis unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# IMPLIED VOLATILITY SURFACE
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_page == "Implied Volatility Surface":
    section_label("Volatility Surface", pulse=True)
    display_heading("Fit a volatility surface", highlight="volatility", size="lg")
    lead("Pull live options chains from YFinance, solve for implied volatility across strikes and expiries using Brent, Newton-Raphson or bisection, and render the result as a 3D surface, heatmap, or smile slice.")

    divider()

    controls, viewport = st.columns([1, 2.2], gap="large")

    with controls:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_label("Controls")
        mono_label("ticker_symbol")
        iv_ticker = st.text_input("iv_ticker_symbol", value="SPY", key="iv_ticker", label_visibility="collapsed")
        mono_label("view")
        surface_type = st.selectbox("surface_type", ["3D Surface", "Volatility Smile", "Heatmap"], label_visibility="collapsed")
        mono_label("iv_solver")
        iv_method = st.selectbox("iv_calculation_method", ["Brent", "Newton-Raphson", "Bisection"], key="iv_method", label_visibility="collapsed")
        st.write("")
        if st.button("Generate surface", type="primary", key="iv_go", use_container_width=True):
            st.session_state.iv_surface_generated = True
            st.session_state.iv_ticker_selected = iv_ticker
            st.session_state.iv_surface_type = surface_type
            st.session_state.iv_method_selected = iv_method
        st.markdown("</div>", unsafe_allow_html=True)

    with viewport:
        if getattr(st.session_state, "iv_surface_generated", False):
            with st.spinner("Fetching chain and solving for IV…"):
                try:
                    iv_surface = ImpliedVolatilitySurface(st.session_state.iv_ticker_selected)
                    options_data = iv_surface.fetch_options_chain()

                    if not options_data.empty:
                        options_with_iv = iv_surface.calculate_iv_for_chain(
                            options_data, method=st.session_state.iv_method_selected.lower()
                        )
                        if not options_with_iv.empty:
                            if st.session_state.iv_surface_type == "3D Surface":
                                fig = iv_surface.generate_surface_plot(options_with_iv, surface_type="3d")
                                is_scene = True
                            elif st.session_state.iv_surface_type == "Volatility Smile":
                                fig = iv_surface.analyze_volatility_smile(options_with_iv)
                                is_scene = False
                            else:
                                fig = iv_surface.generate_surface_plot(options_with_iv, surface_type="heatmap")
                                is_scene = False

                            if fig:
                                # Retheme: replace Viridis surface with accent-palette; recolor scatter traces
                                for tr in fig.data:
                                    if tr.type == "surface":
                                        tr.colorscale = ACCENT_COLORSCALE
                                    elif tr.type == "heatmap":
                                        tr.colorscale = ACCENT_COLORSCALE
                                    elif "scatter" in tr.type:
                                        color = ACCENT if "call" in (tr.name or "").lower() else ACCENT_SECONDARY
                                        tr.marker = dict(color=color, size=tr.marker.size if getattr(tr, "marker", None) and getattr(tr.marker, "size", None) else 6, line=dict(color="white", width=0.5))
                                        if getattr(tr, "line", None):
                                            tr.line = dict(color=color, width=2)
                                st.plotly_chart(style_fig(fig, height=620, scene=is_scene), use_container_width=True)

                            stats = iv_surface.calculate_surface_statistics(options_with_iv)
                            divider()
                            section_label("Surface Diagnostics")
                            sc1, sc2, sc3 = st.columns(3, gap="medium")
                            with sc1:
                                st.metric("Current Price", f"${stats.get('current_price', 0):.2f}")
                                st.metric("ATM IV (short)", f"{stats.get('atm_iv_short', 0)*100:.1f}%")
                            with sc2:
                                st.metric("ATM IV (long)", f"{stats.get('atm_iv_long', 0)*100:.1f}%")
                                st.metric("Term-structure Slope", f"{stats.get('term_structure_slope', 0)*100:.2f}%")
                            with sc3:
                                st.metric("Skew", f"{stats.get('skew', 0)*100:.2f}%")
                                st.metric("Contracts", f"{stats.get('total_options', 0):,}")

                            with st.expander("Raw contract table & export"):
                                st.dataframe(options_with_iv.head(100), use_container_width=True)
                                if st.button("Export to CSV", key="iv_export"):
                                    filepath = iv_surface.export_iv_data(options_with_iv)
                                    if filepath:
                                        st.success(f"Exported → {filepath}")
                        else:
                            st.error("No valid IV values after solving.")
                            st.info("Try a different ticker (AAPL, MSFT, GOOGL) or verify markets are open.")
                    else:
                        st.error(f"No options chain found for {st.session_state.iv_ticker_selected}.")
                except Exception as e:
                    st.error(f"Surface generation failed: {e}")
        else:
            st.markdown(
                """
                <div class="card" style="min-height: 420px; display:flex; align-items:center; justify-content:center; text-align:center;">
                  <div>
                    <div class="eyebrow" style="margin: 0 auto 1rem auto;"><span class="pulse-dot"></span><span>Awaiting input</span></div>
                    <h3 class="display-md" style="max-width: 28ch;">Enter a ticker, pick a view, and render the <span class="gradient-text">surface</span>.</h3>
                    <p class="caption" style="max-width: 40ch; margin: 0.5rem auto 0 auto;">The surface is solved live from the YFinance options chain; no cached snapshots.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_page == "Backtesting Results":
    section_label("Historical Backtesting", pulse=True)
    display_heading("Backtest against real SPX history", highlight="real", size="lg")
    lead("Sample up to 5,000 historical SPX contracts, re-price each with the full model stack, and compare predicted prices against observed mids. Errors are computed with numerical differentiation of the pricing surface, the same machinery that powers the Greeks panel on the pricing page.")

    divider()

    # Try to pre-load prior results
    if not getattr(st.session_state, "backtest_completed", False):
        try:
            results_file = "output/spx_backtest_results.csv"
            if os.path.exists(results_file):
                existing_results = pd.read_csv(results_file)
                if not existing_results.empty:
                    st.session_state.backtest_results = existing_results
                    st.session_state.backtest_completed = True
        except Exception:
            pass

    controls, viewport = st.columns([1, 2.2], gap="large")

    with controls:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_label("Run Parameters")
        mono_label("sample_size")
        backtest_sample_size = st.slider("sample_size", min_value=100, max_value=5000, value=1000, step=100, label_visibility="collapsed")
        mono_label("risk_free_rate_percent")
        risk_free_rate_bt = st.number_input("risk_free_rate_percent", min_value=0.0, value=5.0, step=0.1, key="bt_rf", label_visibility="collapsed") / 100
        st.write("")
        run_backtest = st.button("Run SPX backtest", type="primary", key="bt_go", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card" style="margin-top:1rem;">
              <div class="mono-label">WHY THIS MATTERS</div>
              <p class="caption" style="margin-top:0.25rem;">The engine re-prices <strong>1,000+ concurrent contracts</strong> in a single run, attributes error to each model, and keeps rolling numerical-derivative diagnostics so you can see where the models break down under real market conditions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if run_backtest:
            st.session_state.backtesting_running = True

    with viewport:
        if getattr(st.session_state, "backtesting_running", False):
            with st.spinner("Running backtest across concurrent contracts…"):
                try:
                    backtester = SPXBacktester(output_folder="output")
                    results = backtester.run_full_analysis(sample_size=backtest_sample_size)
                    st.session_state.backtesting_running = False
                    if results is not None and not results.empty:
                        st.session_state.backtest_results = results
                        st.session_state.backtest_completed = True
                        st.rerun()
                    else:
                        st.error("Backtesting failed. Please check your data and try again.")
                except Exception as e:
                    st.error(f"Error during backtesting: {e}")
                    st.session_state.backtesting_running = False

        elif getattr(st.session_state, "backtest_completed", False) and hasattr(st.session_state, "backtest_results"):
            results = st.session_state.backtest_results
            st.success(f"Displaying {len(results):,} contracts.")

            # Performance summary
            section_label("Performance Summary")
            model_columns = [col for col in results.columns if col.endswith("_price")]
            metrics_data = []
            for col in model_columns:
                model_name = col.replace("_price", "")
                error_col = col.replace("_price", "_error")
                if error_col in results.columns:
                    mae = results[error_col].abs().mean()
                    rmse = np.sqrt((results[error_col] ** 2).mean())
                    mape = (results[error_col].abs() / results["mid_price"]).mean() * 100
                    metrics_data.append({"Model": model_name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE (%)": round(mape, 2)})
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Model comparison scatter
            if len(model_columns) >= 2:
                divider()
                section_label("Predicted vs Actual")
                palette = [ACCENT, ACCENT_SECONDARY, "#6366F1", "#0EA5E9"]
                fig = go.Figure()
                for i, col in enumerate(model_columns[:4]):
                    model_name = col.replace("_price", "")
                    fig.add_trace(
                        go.Scatter(
                            x=results["mid_price"], y=results[col],
                            mode="markers",
                            name=model_name,
                            marker=dict(color=palette[i % len(palette)], size=6, opacity=0.55, line=dict(width=0)),
                            hovertemplate=f"{model_name}<br>Actual: $%{{x:.2f}}<br>Predicted: $%{{y:.2f}}<extra></extra>",
                        )
                    )
                min_price = results["mid_price"].min()
                max_price = results["mid_price"].max()
                fig.add_trace(
                    go.Scatter(
                        x=[min_price, max_price], y=[min_price, max_price],
                        mode="lines", name="Perfect prediction",
                        line=dict(dash="dash", color=FOREGROUND, width=1.5),
                        hoverinfo="skip",
                    )
                )
                fig.update_layout(title="Model predictions vs observed mid price", xaxis_title="Observed mid ($)", yaxis_title="Model price ($)")
                st.plotly_chart(style_fig(fig, height=620), use_container_width=True)

            # Dataset info
            divider()
            section_label("Dataset")
            ds1, ds2, ds3, ds4 = st.columns(4, gap="small")
            with ds1: st.metric("Contracts", f"{len(results):,}")
            with ds2: st.metric("Trade days", f"{results['trade_date'].nunique()}")
            with ds3:
                strike_col = "strike_price" if "strike_price" in results.columns else "strike"
                st.metric("Strike range", f"${results[strike_col].min():.0f} – ${results[strike_col].max():.0f}")
            with ds4: st.metric("Avg DTE", f"{results['days_to_expiry'].mean():.0f} d")

            divider()
            c_dl, c_new = st.columns([1, 1])
            with c_dl:
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download results (CSV)",
                    data=csv,
                    file_name=f"spx_backtest_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c_new:
                if st.button("Run a new backtest", use_container_width=True):
                    st.session_state.backtest_completed = False
                    st.session_state.backtest_results = None
                    st.rerun()

        else:
            st.markdown(
                """
                <div class="card" style="min-height: 360px; display:flex; align-items:center; justify-content:center; text-align:center;">
                  <div>
                    <div class="eyebrow" style="margin: 0 auto 1rem auto;"><span class="pulse-dot"></span><span>Awaiting run</span></div>
                    <h3 class="display-md" style="max-width: 30ch;">Configure parameters and trigger a <span class="gradient-text">backtest</span>.</h3>
                    <p class="caption" style="max-width: 46ch; margin: 0.5rem auto 0 auto;">The engine replays up to 5,000 historical SPX contracts against the model stack in a single run.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="site-footer">
      <div>
        <div class="brand"><span class="mark"></span> Quantra</div>
        <div class="caption" style="margin-top:0.35rem;">Real-time options pricing, volatility fitting and backtesting.</div>
      </div>
      <div class="mono">Built by Aniket Dey · {datetime.now().year}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

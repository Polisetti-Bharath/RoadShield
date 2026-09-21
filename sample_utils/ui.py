"""Shared visual styling and reusable UI components for the RoadShield app."""

import streamlit as st

# Damage class -> accent color, tuned for the dark glass theme.
CLASS_COLORS = {
    "Longitudinal Crack": "#60A5FA",  # blue
    "Transverse Crack": "#C084FC",    # violet
    "Alligator Crack": "#FBBF24",     # amber
    "Potholes": "#FB7185",            # rose
}


def inject_base_css() -> None:
    """Injects the shared font, color system, and component styles once per page."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html {
            font-size: 17px;
        }
        html, body, [class*="css"]  {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        h1, h2, h3, .rs-hero h1, .rs-page-header h1, .rs-feature-card h3, .rs-stat .rs-stat-value {
            font-family: 'Bricolage Grotesque', 'Plus Jakarta Sans', sans-serif;
        }

        :root {
            --rs-bg: #0A0E1A;
            --rs-bg-alt: #0F1424;
            --rs-glass: rgba(255, 255, 255, 0.045);
            --rs-glass-border: rgba(255, 255, 255, 0.09);
            --rs-glass-hover: rgba(255, 255, 255, 0.075);
            --rs-ink: #F1F5F9;
            --rs-muted: #8B95AC;
            --rs-accent-1: #6366F1;
            --rs-accent-2: #A855F7;
            --rs-accent-3: #EC4899;
            --rs-accent-4: #22D3EE;
            --rs-gradient: linear-gradient(135deg, #6366F1 0%, #A855F7 55%, #EC4899 100%);
        }

        /* ---- App shell ---- */
        .stApp {
            background:
                radial-gradient(ellipse 900px 500px at 8% -5%, rgba(99, 102, 241, 0.20), transparent 60%),
                radial-gradient(ellipse 800px 500px at 105% 10%, rgba(236, 72, 153, 0.16), transparent 55%),
                radial-gradient(ellipse 700px 600px at 50% 110%, rgba(34, 211, 238, 0.10), transparent 60%),
                var(--rs-bg);
            background-attachment: fixed;
        }

        .block-container {
            padding-top: 2.3rem;
            padding-bottom: 3.5rem;
            max-width: 1150px;
        }

        /* Bump up native widget text, which otherwise stays small */
        [data-testid="stWidgetLabel"] p {
            font-size: 1rem !important;
            font-weight: 600;
        }
        [data-testid="stCaptionContainer"] {
            font-size: 0.9rem !important;
        }
        .stMarkdown p, [data-testid="stMarkdownContainer"] p {
            font-size: 1rem;
            line-height: 1.6;
        }
        div[data-testid="stAlert"] p {
            font-size: 0.97rem !important;
        }
        div.stButton > button, div.stDownloadButton > button {
            font-size: 0.98rem;
            padding: 0.55rem 1.2rem;
        }
        [data-testid="stFileUploaderDropzone"] button {
            font-size: 0.95rem;
        }

        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 8px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.28); }

        @keyframes rsFadeUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes rsFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
        }
        @keyframes rsGlow {
            0%, 100% { opacity: 0.55; }
            50% { opacity: 1; }
        }

        /* ---- Hero ---- */
        .rs-hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(150deg, #151A32 0%, #1B1440 55%, #2A123F 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 2.5rem 2.3rem;
            color: var(--rs-ink);
            margin-bottom: 1.8rem;
            box-shadow: 0 20px 60px -20px rgba(99, 102, 241, 0.35), inset 0 1px 0 rgba(255,255,255,0.06);
            animation: rsFadeUp 0.6s ease both;
        }
        .rs-hero::before {
            content: "";
            position: absolute;
            top: -60px; right: -60px;
            width: 260px; height: 260px;
            background: radial-gradient(circle, rgba(168, 85, 247, 0.45), transparent 70%);
            filter: blur(10px);
            animation: rsFloat 7s ease-in-out infinite;
            pointer-events: none;
        }
        .rs-hero::after {
            content: "";
            position: absolute;
            bottom: -80px; left: -40px;
            width: 240px; height: 240px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.4), transparent 70%);
            filter: blur(10px);
            animation: rsFloat 9s ease-in-out infinite reverse;
            pointer-events: none;
        }
        .rs-hero h1 {
            position: relative;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0 0 0.75rem 0;
            line-height: 1.15;
            letter-spacing: -0.01em;
            background: linear-gradient(90deg, #FFFFFF 20%, #C7D2FE 60%, #F5D0FE 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .rs-hero p {
            position: relative;
            font-size: 1.05rem;
            color: #B6BEDB;
            max-width: 640px;
            margin: 0;
            line-height: 1.6;
        }
        .rs-hero .rs-badge {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.14);
            color: #E0E7FF;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            margin-bottom: 1.1rem;
        }
        .rs-hero .rs-badge .pulse {
            width: 7px; height: 7px; border-radius: 50%;
            background: var(--rs-accent-4);
            box-shadow: 0 0 0 0 rgba(34, 211, 238, 0.6);
            animation: rsGlow 1.8s ease-in-out infinite;
        }

        /* ---- Page header (non-home pages) ---- */
        .rs-page-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.3rem;
            animation: rsFadeUp 0.5s ease both;
        }
        .rs-page-header .rs-icon-badge {
            font-size: 1.8rem;
            background: var(--rs-gradient);
            width: 58px;
            height: 58px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 10px 26px -8px rgba(139, 92, 246, 0.55);
        }
        .rs-page-header h1 {
            font-size: 1.95rem;
            font-weight: 700;
            color: var(--rs-ink);
            margin: 0;
            letter-spacing: -0.01em;
        }
        .rs-page-subtitle {
            color: var(--rs-muted);
            font-size: 1.03rem;
            margin: 0.55rem 0 1.6rem 0;
            line-height: 1.55;
            animation: rsFadeUp 0.6s ease both;
        }

        /* ---- Glass-style wrapper for st.container(key=...) blocks ---- */
        div.st-key-rs-upload-card {
            background: var(--rs-glass);
            border: 1px solid var(--rs-glass-border);
            border-radius: 18px;
            padding: 1.3rem 1.4rem;
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            box-shadow: 0 8px 30px -14px rgba(0,0,0,0.5);
        }

        /* ---- Generic glass card ---- */
        .rs-card {
            position: relative;
            background: var(--rs-glass);
            border: 1px solid var(--rs-glass-border);
            border-radius: 18px;
            padding: 1.5rem 1.6rem;
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            box-shadow: 0 8px 30px -14px rgba(0,0,0,0.5);
            animation: rsFadeUp 0.5s ease both;
        }

        /* ---- Feature grid cards on Home ---- */
        .rs-feature-card {
            position: relative;
            background: var(--rs-glass);
            border: 1px solid var(--rs-glass-border);
            border-radius: 18px;
            padding: 1.6rem 1.5rem;
            height: 100%;
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease, background 0.25s ease;
            box-shadow: 0 8px 30px -14px rgba(0,0,0,0.5);
            animation: rsFadeUp 0.6s ease both;
        }
        .rs-feature-card:hover {
            transform: translateY(-6px);
            background: var(--rs-glass-hover);
            border-color: rgba(255,255,255,0.18);
        }
        .rs-feature-card.glow-indigo:hover { box-shadow: 0 20px 45px -16px rgba(99, 102, 241, 0.55); }
        .rs-feature-card.glow-violet:hover { box-shadow: 0 20px 45px -16px rgba(168, 85, 247, 0.55); }
        .rs-feature-card.glow-pink:hover   { box-shadow: 0 20px 45px -16px rgba(236, 72, 153, 0.55); }
        .rs-feature-card .rs-feature-icon {
            font-size: 1.6rem;
            width: 52px;
            height: 52px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.9rem;
        }
        .rs-feature-card.glow-indigo .rs-feature-icon { background: linear-gradient(135deg, #6366F1, #4338CA); }
        .rs-feature-card.glow-violet .rs-feature-icon { background: linear-gradient(135deg, #A855F7, #7E22CE); }
        .rs-feature-card.glow-pink .rs-feature-icon   { background: linear-gradient(135deg, #EC4899, #BE185D); }
        .rs-feature-card h3 {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--rs-ink);
            margin: 0 0 0.45rem 0;
        }
        .rs-feature-card p {
            font-size: 0.93rem;
            color: var(--rs-muted);
            line-height: 1.55;
            margin: 0;
        }

        /* ---- Damage class pill legend ---- */
        .rs-pill-row { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 0.7rem; }
        .rs-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            font-size: 0.87rem;
            font-weight: 600;
            color: var(--rs-ink);
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .rs-pill .dot {
            width: 9px; height: 9px; border-radius: 50%;
        }

        /* ---- Section label ---- */
        .rs-section-label {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--rs-muted);
            margin-bottom: 0.6rem;
        }

        /* ---- Metric-style stat cards ---- */
        .rs-stat {
            background: var(--rs-glass);
            border: 1px solid var(--rs-glass-border);
            border-radius: 16px;
            padding: 1.05rem 1rem;
            text-align: center;
            backdrop-filter: blur(14px);
            animation: rsFadeUp 0.5s ease both;
        }
        .rs-stat .rs-stat-value {
            font-size: 1.75rem;
            font-weight: 700;
        }
        .rs-stat .rs-stat-label {
            font-size: 0.82rem;
            color: var(--rs-muted);
            font-weight: 600;
            margin-top: 0.25rem;
        }

        /* ---- Callout tip ---- */
        .rs-tip {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            background: var(--rs-glass);
            border: 1px solid var(--rs-glass-border);
            border-left: 3px solid var(--rs-accent-4);
            border-radius: 14px;
            padding: 0.95rem 1.15rem;
            backdrop-filter: blur(14px);
            animation: rsFadeUp 0.5s ease both;
        }
        .rs-tip .rs-tip-icon { font-size: 1.3rem; }
        .rs-tip .rs-tip-text {
            font-size: 0.98rem;
            font-weight: 600;
            color: var(--rs-ink);
        }

        /* ---- Native Streamlit widget restyling ---- */
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 12px;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            color: var(--rs-ink);
            transition: all 0.2s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            border-color: rgba(255,255,255,0.3);
            background: rgba(255,255,255,0.08);
            transform: translateY(-1px);
        }
        div.stButton > button[kind="primary"], div.stDownloadButton > button {
            background: var(--rs-gradient);
            border: none;
            box-shadow: 0 8px 24px -8px rgba(139, 92, 246, 0.55);
        }
        div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button:hover {
            filter: brightness(1.12);
            box-shadow: 0 10px 28px -6px rgba(139, 92, 246, 0.7);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--rs-glass);
            border: 1.5px dashed rgba(255,255,255,0.18);
            border-radius: 14px;
            transition: border-color 0.2s ease, background 0.2s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--rs-accent-4);
            background: rgba(34, 211, 238, 0.06);
        }

        div[data-baseweb="slider"] > div > div > div { background: var(--rs-gradient); }

        div[data-testid="stExpander"] {
            background: var(--rs-glass);
            border: 1px solid var(--rs-glass-border);
            border-radius: 14px;
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid var(--rs-glass-border);
            backdrop-filter: blur(10px);
        }

        /* ---- Footer ---- */
        .rs-footer {
            margin-top: 3rem;
            padding-top: 1.4rem;
            border-top: 1px solid var(--rs-glass-border);
            color: var(--rs-muted);
            font-size: 0.85rem;
            text-align: center;
        }
        .rs-footer .rs-footer-gradient {
            background: var(--rs-gradient);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-weight: 700;
        }

        /* ---- Sidebar polish ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D1122 0%, #0A0E1A 100%);
            border-right: 1px solid rgba(255,255,255,0.07);
        }
        [data-testid="stSidebarNavLink"] {
            border-radius: 10px !important;
            margin: 2px 0.5rem;
            padding: 0.45rem 0.7rem !important;
            font-size: 0.98rem !important;
            transition: background 0.2s ease;
        }
        [data-testid="stSidebarNavLink"]:hover {
            background: rgba(255,255,255,0.06);
        }
        [data-testid="stSidebarNavLink"][aria-current="page"] {
            background: linear-gradient(90deg, rgba(99,102,241,0.25), rgba(236,72,153,0.12));
            border-left: 3px solid var(--rs-accent-1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(badge: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="rs-hero">
            <span class="rs-badge"><span class="pulse"></span>{badge}</span>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="rs-page-header">
            <div class="rs-icon-badge">{icon}</div>
            <h1>{title}</h1>
        </div>
        <p class="rs-page-subtitle">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


def render_class_legend() -> None:
    pills = "".join(
        f'<span class="rs-pill"><span class="dot" style="background:{color}"></span>{label}</span>'
        for label, color in CLASS_COLORS.items()
    )
    st.markdown(f'<div class="rs-pill-row">{pills}</div>', unsafe_allow_html=True)


def render_tip(icon: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="rs-tip">
            <span class="rs-tip-icon">{icon}</span>
            <span class="rs-tip-text">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="rs-footer">
            <span class="rs-footer-gradient">RoadShield</span> &middot; Powered by YOLOv8 &middot; Trained on the CRDDC2022 dataset
        </div>
        """,
        unsafe_allow_html=True,
    )

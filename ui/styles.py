import streamlit as st


def load_global_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #07111f;
            --bg-1: #0c1830;
            --bg-2: #101d39;
            --card: rgba(13, 25, 49, 0.82);
            --card-border: rgba(99, 179, 237, 0.18);
            --text-main: #e7f0ff;
            --text-soft: #8ea6cf;
            --cyan: #3be7ff;
            --blue: #4f7cff;
            --red: #ff4d5a;
            --glow: 0 0 0 1px rgba(59,231,255,0.08), 0 18px 60px rgba(16, 42, 89, 0.35);
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(53,120,255,0.18), transparent 30%),
                radial-gradient(circle at 85% 15%, rgba(59,231,255,0.10), transparent 25%),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 38%, #060d19 100%);
            color: var(--text-main);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
            background-size: 28px 28px;
            mask-image: linear-gradient(180deg, rgba(255,255,255,0.35), transparent 85%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6, 15, 30, 0.98) 0%, rgba(8, 18, 36, 0.96) 100%);
            border-right: 1px solid rgba(79, 124, 255, 0.16);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }

        .sidebar-shell {
            padding: 10px 4px 18px 4px;
        }

        .sidebar-logo {
            font-size: 34px;
            line-height: 1;
            font-weight: 900;
            letter-spacing: 0.02em;
            color: #ffffff;
            margin-bottom: 6px;
        }

        .sidebar-logo-accent {
            display: inline-block;
            font-size: 14px;
            font-weight: 800;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--cyan);
            margin-bottom: 18px;
        }

        .sidebar-subtitle {
            font-size: 12px;
            line-height: 1.75;
            color: #9fb4d6;
            margin-bottom: 18px;
        }

        .nav-group-label {
            font-size: 11px;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: rgba(140, 167, 207, 0.72);
            margin: 6px 0 10px 0;
        }

        .sidebar-footnote {
            margin-top: 18px;
            padding-top: 16px;
            border-top: 1px solid rgba(79, 124, 255, 0.12);
            font-size: 12px;
            line-height: 1.8;
            color: #8ea6cf;
        }

        section[data-testid="stSidebar"] .stButton > button {
            min-height: 48px;
            border-radius: 14px;
            font-weight: 700;
            font-size: 14px;
            color: #dfe9ff;
            background: linear-gradient(180deg, rgba(18, 31, 56, 0.96) 0%, rgba(11, 21, 40, 0.96) 100%);
            border: 1px solid rgba(100, 170, 255, 0.18);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
            transition: all 0.16s ease;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            color: #ffffff;
            border-color: rgba(59,231,255,0.55);
            box-shadow: 0 0 0 1px rgba(59,231,255,0.14), 0 0 24px rgba(59,231,255,0.16);
            transform: translateY(-1px);
        }

        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            color: #04121f;
            background: linear-gradient(135deg, var(--cyan) 0%, #8ff5ff 45%, #4f7cff 100%);
            border: none;
            box-shadow: 0 8px 28px rgba(59,231,255,0.28);
        }

        .page-title {
            font-size: 42px;
            font-weight: 900;
            letter-spacing: 0.01em;
            color: #f4f8ff;
            margin-bottom: 8px;
        }

        .page-desc {
            font-size: 14px;
            line-height: 1.8;
            color: var(--text-soft);
            margin-bottom: 18px;
            max-width: 980px;
        }

        h1, h2, h3, h4, label, p, li, span, div {
            color: inherit;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(13, 25, 49, 0.88) 0%, rgba(9, 19, 37, 0.88) 100%);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 18px 18px 16px 18px;
            box-shadow: var(--glow);
        }

        [data-testid="stMetricLabel"] {
            color: #8ea6cf;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff;
        }

        [data-testid="stExpander"], .stAlert, .stTabs, .stDataFrame, .stTable {
            border-radius: 18px;
        }

        [data-testid="stAlert"] {
            background: rgba(16, 30, 58, 0.82);
            border: 1px solid rgba(95, 161, 255, 0.18);
            color: #dce8ff;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(11, 22, 44, 0.72);
            border: 1px solid rgba(100, 170, 255, 0.16);
            border-radius: 14px;
            color: #9bb5dc;
            padding: 10px 16px;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(59,231,255,0.16), rgba(79,124,255,0.18));
            color: #ffffff !important;
            border-color: rgba(59,231,255,0.45);
        }

        .stButton > button {
            border-radius: 14px;
            font-weight: 700;
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
        }

        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stSelectbox [data-baseweb="select"] > div,
        .stNumberInput input {
            background: rgba(11, 22, 44, 0.82) !important;
            color: #eaf2ff !important;
            border: 1px solid rgba(96, 168, 255, 0.18) !important;
            border-radius: 12px !important;
        }

        .stMarkdown, .stCaption, .stText, .st-emotion-cache-10trblm {
            color: inherit;
        }

        .stDataFrame, [data-testid="stDataFrame"] {
            background: rgba(11, 22, 44, 0.78);
            border: 1px solid rgba(96, 168, 255, 0.14);
            border-radius: 18px;
            box-shadow: var(--glow);
        }

        .filter-card,
        .intel-card,
        .intel-detail-card,
        .hot-card {
            position: relative;
            background: linear-gradient(180deg, rgba(13, 25, 49, 0.92) 0%, rgba(8, 18, 36, 0.88) 100%);
            border: 1px solid rgba(96, 168, 255, 0.16);
            border-radius: 20px;
            padding: 12px 14px;
            box-shadow: var(--glow);
            overflow: hidden;
            animation: panelFadeIn 0.35s ease;
        }

        .filter-card::before,
        .intel-card::before,
        .intel-detail-card::before,
        .hot-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, rgba(59,231,255,0.06), transparent 36%, transparent 64%, rgba(79,124,255,0.08));
            pointer-events: none;
        }

        .filter-card:hover,
        .intel-card:hover,
        .intel-detail-card:hover,
        .hot-card:hover {
            border-color: rgba(59,231,255,0.28);
            box-shadow: 0 0 0 1px rgba(59,231,255,0.08), 0 22px 60px rgba(16, 42, 89, 0.4);
            transform: translateY(-1px);
        }

        .section-label {
            margin: 8px 0 10px 0;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--cyan);
        }

        .mini-filter-title {
            margin-bottom: 6px;
            font-size: 13px;
            font-weight: 700;
            color: #dce8ff;
        }

        .intel-empty-card {
            min-height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .intel-detail-title {
            font-size: 20px;
            font-weight: 800;
            color: #f5f8ff;
            margin-bottom: 8px;
        }

        .intel-detail-empty {
            color: #9eb2d6;
            line-height: 1.8;
        }

        .hot-grid-title {
            margin: 10px 0 12px 0;
            font-size: 15px;
            font-weight: 800;
            color: #f5f8ff;
        }

        .hot-card {
            min-height: 470px;
            padding-top: 14px;
        }

        .hot-card-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 10px 12px;
            border-radius: 14px;
            color: #0f172a;
            font-weight: 800;
        }

        .hot-card-title {
            font-size: 18px;
            font-weight: 900;
            letter-spacing: 0.01em;
        }

        .hot-card-sub {
            font-size: 11px;
            opacity: 0.85;
        }

        .theme-red .hot-card-head {
            background: linear-gradient(135deg, #ffd0d8 0%, #ffb5c5 100%);
        }

        .theme-blue .hot-card-head {
            background: linear-gradient(135deg, #cbe7ff 0%, #b4d9ff 100%);
        }

        .theme-sky .hot-card-head {
            background: linear-gradient(135deg, #d7f2ff 0%, #bfe9ff 100%);
        }

        .theme-indigo .hot-card-head {
            background: linear-gradient(135deg, #d9ddff 0%, #c7d2fe 100%);
        }

        .theme-gold .hot-card-head {
            background: linear-gradient(135deg, #ffe9b8 0%, #ffd66b 100%);
        }

        .hot-card div[data-testid="stButton"] > button {
            min-height: 38px;
            margin-bottom: 8px;
            text-align: left;
            justify-content: flex-start;
            background: rgba(245, 248, 255, 0.98) !important;
            color: #12243f !important;
            border: 1px solid rgba(181, 194, 214, 0.55) !important;
            box-shadow: none !important;
            font-size: 13px;
            font-weight: 700;
        }

        .hot-card div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, rgba(255,99,132,0.18), rgba(255,255,255,0.98)) !important;
            color: #0c2140 !important;
            border: 1px solid rgba(255,99,132,0.45) !important;
            box-shadow: 0 8px 18px rgba(255,99,132,0.12) !important;
        }

        .hot-card div[data-testid="stButton"] > button:hover {
            background: linear-gradient(135deg, #ffffff, #dff4ff) !important;
            color: #08182c !important;
            box-shadow: 0 10px 22px rgba(59,231,255,0.14) !important;
        }

        .hot-tag {
            height: 38px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            background: rgba(13, 25, 49, 0.62);
            color: #c6d4ec;
            font-size: 12px;
            font-weight: 700;
            border: 1px solid rgba(96, 168, 255, 0.14);
            margin-bottom: 8px;
        }

        @keyframes panelFadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        hr {
            border-color: rgba(96, 168, 255, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

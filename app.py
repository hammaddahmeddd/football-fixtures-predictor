
"""
Football Fixtures Result Predictor
Modern Streamlit Dashboard

Run:
    python -m streamlit run app.py
"""

import streamlit as st

from dashboard import (
    overview,
    match_predictions,
    model_comparison,
    goal_analysis,
    season_table,
    strategy_maker,
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Football Fixtures Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# CUSTOM STYLES
# ==========================================

st.markdown(
    """
    <style>

    /* Main content width and spacing */

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    /* Main headings */

    h1 {
        font-size: 2.15rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.8px;
    }

    h2 {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 650 !important;
    }

    /* Metric cards */

    div[data-testid="stMetric"] {
        background: #162238;
        border: 1px solid #30425C;
        border-radius: 14px;
        padding: 20px;
        min-height: 115px;
    }

    div[data-testid="stMetricLabel"] {
        color: #BAC8DA !important;
        font-size: 0.9rem;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.8rem;
        font-weight: 750;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #2B3D54;
    }

    section[data-testid="stSidebar"] .stRadio label {
        padding-top: 5px;
        padding-bottom: 5px;
    }

    /* Buttons */

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* Dataframes */

    div[data-testid="stDataFrame"] {
        border: 1px solid #30425C;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Informational alert */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Dividers */

    hr {
        border-color: #30425C !important;
        opacity: 0.65;
    }

    /* Mobile responsive */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        h1 {
            font-size: 1.7rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.4rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# PAGE NAVIGATION
# ==========================================

PAGES = {
    "Overview": overview,
    "Match Predictions": match_predictions,
    "Model Comparison": model_comparison,
    "Goal Analysis": goal_analysis,
    "Season Table": season_table,
    "Strategy Maker": strategy_maker,
}

PAGE_ICONS = {
    "Overview": "📊",
    "Match Predictions": "⚽",
    "Model Comparison": "📈",
    "Goal Analysis": "🎯",
    "Season Table": "🏆",
    "Strategy Maker": "🧠",
}


def main():

    # ======================================
    # SIDEBAR
    # ======================================

    with st.sidebar:

        st.title("⚽ Football Predictor")

        st.caption(
            "PREMIER LEAGUE 2026/27"
        )

        st.divider()

        st.markdown("**NAVIGATION**")

        selected_page = st.radio(
            "Select Page",
            list(PAGES.keys()),
            format_func=lambda page: (
                f"{PAGE_ICONS[page]}  {page}"
            ),
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("**PROJECT INFORMATION**")

        st.caption(
            "Football Fixtures Result Predictor"
        )

        st.caption(
            "Machine Learning University Project"
        )

        st.markdown(
            """
            **Implemented Models**

            KNN · SVM · Decision Tree

            Random Forest · Regression

            K-Means · Q-Learning
            """
        )

        st.divider()

        if st.button(
            "🔄 Refresh Dashboard",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.rerun()

        st.caption(
            "Refresh updates locally saved data. "
            "It does not automatically retrain models."
        )

    # ======================================
    # DISPLAY SELECTED PAGE
    # ======================================

    PAGES[selected_page].render()


if __name__ == "__main__":
    main()

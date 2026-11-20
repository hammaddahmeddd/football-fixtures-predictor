
"""
Football Fixtures Result Predictor
Dashboard: Overview Page
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    RAW_DATA_DIR,
    PREDICTIONS_DIR,
    OUTPUTS_DIR,
)


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def read_csv(path):
    """
    Load a CSV file if it exists.
    Return None if the file is missing.
    """

    path = Path(path)

    if not path.exists():
        return None

    return pd.read_csv(path)


# ==========================================
# MAIN OVERVIEW PAGE
# ==========================================

def render():

    st.title("⚽ Football Fixtures Predictor")

    st.write(
        "A machine learning system for predicting "
        "Premier League match outcomes, total goals, "
        "and full-season league standings."
    )

    st.caption(
        "Premier League 2026/27 | "
        "Machine Learning University Project"
    )

    st.divider()

    # ======================================
    # LOAD PROJECT DATA
    # ======================================

    historical = read_csv(
        RAW_DATA_DIR / "premier_league_raw.csv"
    )

    current = read_csv(
        RAW_DATA_DIR / "premier_league_2026_27.csv"
    )

    weekly = read_csv(
        PREDICTIONS_DIR / "weekly_predictions.csv"
    )

    season = read_csv(
        OUTPUTS_DIR
        / "metrics"
        / "season_simulation_summary.csv"
    )

    # ======================================
    # PROJECT SUMMARY METRICS
    # ======================================

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Historical Matches",
            f"{len(historical):,}"
            if historical is not None
            else "N/A"
        )

    with col2:
        st.metric(
            "Current Season Matches",
            f"{len(current):,}"
            if current is not None
            else "N/A"
        )

    with col3:
        st.metric(
            "Weekly Fixtures",
            len(weekly)
            if weekly is not None
            else "N/A"
        )

    with col4:
        st.metric(
            "Teams Simulated",
            len(season)
            if season is not None
            else "N/A"
        )

    st.divider()

    # ======================================
    # PROJECT OBJECTIVES
    # ======================================

    left, right = st.columns(2)

    with left:

        st.subheader("🎯 Project Objectives")

        st.markdown(
            """
            - Predict Home Win, Draw, and Away Win outcomes
            - Estimate total goals scored in matches
            - Predict Over/Under 2.5 goals
            - Compare multiple classification algorithms
            - Apply K-Means clustering to team profiles
            - Use Q-Learning for prediction model selection
            - Simulate the remaining Premier League season
            - Estimate title, top-four, and relegation probabilities
            """
        )

    # ======================================
    # ALGORITHMS USED
    # ======================================

    with right:

        st.subheader("🧠 Machine Learning Algorithms")

        algorithms = pd.DataFrame({
            "Category": [
                "Classification",
                "Classification",
                "Classification",
                "Ensemble",
                "Regression",
                "Clustering",
                "Reinforcement Learning",
                "Simulation",
            ],
            "Algorithm": [
                "K-Nearest Neighbors",
                "Support Vector Machine",
                "Decision Tree",
                "Random Forest",
                "Random Forest Regressor",
                "K-Means",
                "Q-Learning",
                "Monte Carlo Simulation",
            ]
        })

        st.dataframe(
            algorithms,
            hide_index=True,
            width="stretch"
        )

    st.divider()

    # ======================================
    # PROJECTED LEAGUE LEADERS
    # ======================================

    st.subheader("🏆 Projected League Leaders")

    if season is not None and not season.empty:

        leaders = season.sort_values(
            "ProjectedPosition"
        ).head(5).copy()

        # Convert probability fractions to
        # percentages for display.
        #
        # Example:
        # 0.543 becomes 54.3%

        leaders["TitleProbability"] = (
            leaders["TitleProbability"] * 100
        )

        leaders["Top4Probability"] = (
            leaders["Top4Probability"] * 100
        )

        display = leaders[
            [
                "ProjectedPosition",
                "Team",
                "ExpectedPoints",
                "TitleProbability",
                "Top4Probability",
            ]
        ].copy()

        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "ProjectedPosition":
                    st.column_config.NumberColumn(
                        "Position",
                        format="%d"
                    ),

                "ExpectedPoints":
                    st.column_config.NumberColumn(
                        "Expected Points",
                        format="%.1f"
                    ),

                "TitleProbability":
                    st.column_config.NumberColumn(
                        "Title Chance",
                        format="%.1f%%"
                    ),

                "Top4Probability":
                    st.column_config.NumberColumn(
                        "Top 4 Chance",
                        format="%.1f%%"
                    ),
            }
        )

        # ==================================
        # TITLE PROBABILITY CHART
        # ==================================

        st.subheader("📊 Title Winning Probabilities")

        title_chart = leaders[
            [
                "Team",
                "TitleProbability"
            ]
        ].set_index("Team")

        st.bar_chart(
            title_chart,
            y_label="Title Probability (%)",
            width="stretch"
        )

        # ==================================
        # EXPECTED FINAL POINTS
        # ==================================

        st.subheader("📈 Expected Final Points")

        points_chart = leaders[
            [
                "Team",
                "ExpectedPoints"
            ]
        ].set_index("Team")

        st.bar_chart(
            points_chart,
            y_label="Expected Points",
            width="stretch"
        )

        # ==================================
        # LEAGUE LEADER HIGHLIGHTS
        # ==================================

        st.subheader("Season Projection Highlights")

        champion = season.loc[
            season["TitleProbability"].idxmax()
        ]

        top_points = season.loc[
            season["ExpectedPoints"].idxmax()
        ]

        high_relegation = season.loc[
            season["RelegationProbability"].idxmax()
        ]

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Highest Title Chance",
                champion["Team"],
                f"{champion['TitleProbability']:.1%}"
            )

        with b:
            st.metric(
                "Highest Expected Points",
                top_points["Team"],
                f"{top_points['ExpectedPoints']:.1f} pts",
                delta_color="off"
            )

        with c:
            st.metric(
                "Highest Relegation Risk",
                high_relegation["Team"],
                f"{high_relegation['RelegationProbability']:.1%}",
                delta_color="off"
            )

        st.caption(
            "Projections are based on the saved "
            "Monte Carlo season simulation. "
            "They represent model estimates, "
            "not guaranteed future outcomes."
        )

    else:

        st.info(
            "Season simulation results are not "
            "available yet. Run the following "
            "command in your terminal:"
        )

        st.code(
            "python -m src.season_simulation",
            language="powershell"
        )

    st.divider()

    # ======================================
    # COMPLETE PROJECT WORKFLOW
    # ======================================

    st.subheader("🔄 Project Workflow")

    st.code(
        "Historical Football Data + Live Fixtures\n"
        "                  |\n"
        "                  v\n"
        "          Data Preprocessing\n"
        "                  |\n"
        "                  v\n"
        "          Feature Engineering\n"
        "                  |\n"
        "                  v\n"
        "     Classification + Regression\n"
        "                  |\n"
        "                  v\n"
        "     K-Means + Q-Learning Models\n"
        "                  |\n"
        "                  v\n"
        "          Weekly Predictions\n"
        "                  |\n"
        "                  v\n"
        "       Monte Carlo Simulation\n"
        "                  |\n"
        "                  v\n"
        "         Streamlit Dashboard",
        language="text"
    )

    # ======================================
    # FOOTER
    # ======================================

    st.divider()

    st.caption(
        "Football Fixtures Result Predictor | "
        "Premier League 2026/27 | "
        "University Machine Learning Project"
    )



"""
Dashboard: Weekly Match Predictions
"""

import pandas as pd
import streamlit as st

from config import PREDICTIONS_DIR


RESULT_NAMES = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}


@st.cache_data
def load_predictions():

    path = (
        PREDICTIONS_DIR
        / "weekly_predictions.csv"
    )

    if not path.exists():
        return None

    return pd.read_csv(path)


def render():

    st.title("âš½ Match Predictions")

    st.write(
        "Explore upcoming Premier League fixtures, "
        "predicted results and model decisions."
    )

    df = load_predictions()

    if df is None or df.empty:

        st.warning(
            "Weekly predictions are unavailable. "
            "Run python -m src.predictions."
        )
        return

    df = df.copy()

    # ======================================
    # SUMMARY CARDS
    # ======================================

    home_wins = int(
        (df["PredictedResult"] == "H").sum()
    )

    draws = int(
        (df["PredictedResult"] == "D").sum()
    )

    away_wins = int(
        (df["PredictedResult"] == "A").sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Upcoming Matches",
        len(df)
    )

    col2.metric(
        "Home Wins",
        home_wins
    )

    col3.metric(
        "Draws",
        draws
    )

    col4.metric(
        "Away Wins",
        away_wins
    )

    st.divider()

    # ======================================
    # FIXTURE TABLE
    # ======================================

    st.subheader("Upcoming Fixtures")

    display = df[
        [
            "HomeTeam",
            "AwayTeam",
            "ResultLabel",
            "PredictedTotalGoals",
            "OverUnder2_5",
            "SelectedModel",
            "SelectedConfidence",
        ]
    ].copy()

    display["SelectedConfidence"] *= 100

    display = display.rename(
        columns={
            "HomeTeam": "Home Team",
            "AwayTeam": "Away Team",
            "ResultLabel": "Prediction",
            "PredictedTotalGoals": "Expected Goals",
            "OverUnder2_5": "Over / Under 2.5",
            "SelectedModel": "Selected Model",
            "SelectedConfidence": "Confidence (%)",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Expected Goals":
                st.column_config.NumberColumn(
                    "Expected Goals",
                    format="%.2f"
                ),
            "Confidence (%)":
                st.column_config.ProgressColumn(
                    "Confidence (%)",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
        },
    )

    st.download_button(
        "Download All Predictions",
        df.to_csv(index=False),
        file_name="weekly_predictions.csv",
        mime="text/csv",
        width="content",
    )

    st.divider()

    # ======================================
    # INDIVIDUAL MATCH ANALYSIS
    # ======================================

    st.subheader("Match Analysis")

    fixture_names = [
        f"{row.HomeTeam} vs {row.AwayTeam}"
        for row in df.itertuples()
    ]

    selected_index = st.selectbox(
        "Choose a fixture to analyse",
        options=range(len(df)),
        format_func=lambda index: (
            fixture_names[index]
        ),
    )

    match = df.iloc[selected_index]

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    st.markdown(
        f"### {home_team} vs {away_team}"
    )

    # ======================================
    # PROMINENT OUTCOME CARD
    # ======================================

    result = match["ResultLabel"]

    with st.container(border=True):

        st.caption(
            "PREDICTED MATCH OUTCOME"
        )

        st.markdown(
            f"## ðŸ… {result}"
        )

        st.write(
            f"Q-Learning selected the "
            f"**{match['SelectedModel']}** "
            "classifier for this fixture."
        )

        st.caption(
            "This is a model prediction, "
            "not a confirmed result."
        )

    # ======================================
    # MATCH METRICS
    # ======================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Expected Goals",
        f"{match['PredictedTotalGoals']:.2f}"
    )

    col2.metric(
        "Goals Prediction",
        f"{match['OverUnder2_5']} 2.5"
    )

    col3.metric(
        "Model Confidence",
        f"{match['SelectedConfidence']:.1%}"
    )

    st.divider()

    # ======================================
    # CLASSIFIER COMPARISON
    # ======================================

    st.subheader("Classifier Predictions")

    classifiers = {
        "KNN": (
            "KNN_Prediction",
            "KNN_Confidence",
        ),
        "SVM": (
            "SVM_Prediction",
            "SVM_Confidence",
        ),
        "Decision Tree": (
            "Decision_Tree_Prediction",
            "Decision_Tree_Confidence",
        ),
        "Random Forest": (
            "Random_Forest_Prediction",
            "Random_Forest_Confidence",
        ),
    }

    rows = []

    for name, (pred_col, conf_col) in (
        classifiers.items()
    ):

        if pred_col not in df.columns:
            continue

        code = match[pred_col]

        confidence = (
            float(match[conf_col]) * 100
            if conf_col in df.columns
            else None
        )

        rows.append({
            "Classifier": name,
            "Predicted Result":
                RESULT_NAMES.get(
                    code,
                    str(code)
                ),
            "Confidence (%)": confidence,
            "Selected by RL": (
                "Yes"
                if name == match["SelectedModel"]
                else "No"
            ),
        })

    if rows:

        comparison = pd.DataFrame(
            rows
        )

        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "Confidence (%)":
                    st.column_config.ProgressColumn(
                        "Confidence (%)",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%"
                    ),
            }
        )

    # ======================================
    # STRATEGY EXPLANATION
    # ======================================

    st.subheader("Why This Model?")

    if "StrategyExplanation" in df.columns:

        st.info(
            match["StrategyExplanation"]
        )

    else:

        st.write(
            "The Q-Learning policy selected "
            f"{match['SelectedModel']}."
        )

    st.caption(
        "Model confidence is the classifier's "
        "estimated probability for its selected "
        "class. Its accuracy as a probability "
        "estimate has not been independently verified."
    )


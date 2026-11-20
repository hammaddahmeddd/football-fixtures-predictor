
"""
Dashboard: Prediction History and Live Accuracy
"""

import pandas as pd
import streamlit as st

from config import PREDICTIONS_DIR, OUTPUTS_DIR


HISTORY_PATH = (
    PREDICTIONS_DIR / "prediction_history.csv"
)

SUMMARY_PATH = (
    OUTPUTS_DIR / "metrics" /
    "live_accuracy_summary.csv"
)

RESULT_NAMES = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}


def format_accuracy(value):

    if pd.isna(value):
        return "N/A"

    return f"{float(value):.1%}"


def render():

    st.title("📊 Prediction History")

    st.write(
        "Track original pre-match predictions "
        "and compare them against actual results."
    )

    if not HISTORY_PATH.exists():

        st.warning(
            "Prediction history is not available."
        )

        return

    history = pd.read_csv(HISTORY_PATH)

    if history.empty:

        st.info(
            "No predictions have been registered."
        )

        return

    finished = history[
        history["Status"] == "Finished"
    ].copy()

    pending = history[
        history["Status"] == "Pending"
    ].copy()

    result_accuracy = (
        finished["ResultCorrect"].mean()
        if not finished.empty
        else float("nan")
    )

    over_under_accuracy = (
        finished["OverUnderCorrect"].mean()
        if not finished.empty
        else float("nan")
    )

    goal_mae = (
        finished["AbsoluteGoalError"].mean()
        if not finished.empty
        else float("nan")
    )

    st.subheader("Live Performance Overview")

    a, b, c = st.columns(3)

    a.metric(
        "Registered Predictions",
        len(history)
    )

    b.metric(
        "Completed Matches",
        len(finished)
    )

    c.metric(
        "Pending Matches",
        len(pending)
    )

    a, b, c = st.columns(3)

    a.metric(
        "Match Result Accuracy",
        format_accuracy(result_accuracy)
    )

    b.metric(
        "Over / Under Accuracy",
        format_accuracy(over_under_accuracy)
    )

    c.metric(
        "Goal Prediction MAE",
        "N/A"
        if pd.isna(goal_mae)
        else f"{goal_mae:.2f}"
    )

    if finished.empty:

        st.info(
            "Live accuracy will appear once "
            "registered matches are completed."
        )

    else:

        st.caption(
            f"Live performance is based on "
            f"{len(finished)} completed fixtures."
        )

    st.divider()

    st.subheader("Prediction Records")

    col1, col2 = st.columns(2)

    with col1:

        status = st.selectbox(
            "Filter by status",
            ["All", "Pending", "Finished", "Invalid"]
        )

    with col2:

        matchday = st.selectbox(
            "Filter by matchday",
            ["All"] + sorted(
                history["Matchday"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    filtered = history.copy()

    if status != "All":

        filtered = filtered[
            filtered["Status"] == status
        ]

    if matchday != "All":

        filtered = filtered[
            filtered["Matchday"] == matchday
        ]

    filtered["PredictedResult"] = (
        filtered["PredictedResult"]
        .map(RESULT_NAMES)
    )

    filtered["ActualResult"] = (
        filtered["ActualResult"]
        .map(RESULT_NAMES)
    )

    filtered["SelectedConfidence"] *= 100

    filtered["ProbOver2_5"] *= 100

    filtered["PredictionStatus"] = "Awaiting Result"

    filtered.loc[
        filtered["ResultCorrect"] == 1,
        "PredictionStatus"
    ] = "Correct"

    filtered.loc[
        filtered["ResultCorrect"] == 0,
        "PredictionStatus"
    ] = "Incorrect"

    filtered.loc[
        filtered["Status"] == "Invalid",
        "PredictionStatus"
    ] = "Invalid"

    display_columns = [
        "Matchday",
        "HomeTeam",
        "AwayTeam",
        "PredictedResult",
        "ActualResult",
        "SelectedModel",
        "SelectedConfidence",
        "PredictedTotalGoals",
        "ActualTotalGoals",
        "ProbOver2_5",
        "PredictionStatus",
    ]

    st.dataframe(
        filtered[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "Matchday": "Matchday",
            "HomeTeam": "Home Team",
            "AwayTeam": "Away Team",
            "PredictedResult": "Prediction",
            "ActualResult": "Actual Result",
            "SelectedModel": "Selected Model",
            "SelectedConfidence":
                st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
            "PredictedTotalGoals":
                st.column_config.NumberColumn(
                    "Expected Goals",
                    format="%.2f"
                ),
            "ActualTotalGoals": "Actual Goals",
            "ProbOver2_5":
                st.column_config.ProgressColumn(
                    "Over 2.5 Chance",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
            "PredictionStatus": "Performance",
        }
    )

    st.download_button(
        "Download Prediction History",
        history.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv",
        width="content"
    )

    st.divider()

    if not finished.empty:

        st.subheader("Cumulative Prediction Accuracy")

        finished["KickoffUTC"] = pd.to_datetime(
            finished["KickoffUTC"],
            utc=True
        )

        finished = finished.sort_values(
            ["KickoffUTC", "MatchID"]
        ).reset_index(drop=True)

        finished["MatchNumber"] = (
            finished.index + 1
        )

        finished["Result Accuracy (%)"] = (
            finished["ResultCorrect"]
            .expanding()
            .mean() * 100
        )

        finished["Over / Under Accuracy (%)"] = (
            finished["OverUnderCorrect"]
            .expanding()
            .mean() * 100
        )

        st.line_chart(
            finished.set_index("MatchNumber")[
                [
                    "Result Accuracy (%)",
                    "Over / Under Accuracy (%)",
                ]
            ],
            y_label="Accuracy (%)"
        )

        st.subheader("Selected Model Performance")

        model_performance = (
            finished.groupby("SelectedModel")
            .agg(
                Matches=("ResultCorrect", "count"),
                Correct=("ResultCorrect", "sum"),
                Accuracy=("ResultCorrect", "mean")
            )
            .reset_index()
        )

        model_performance["Accuracy"] *= 100

        st.dataframe(
            model_performance,
            hide_index=True,
            width="stretch",
            column_config={
                "Accuracy":
                    st.column_config.NumberColumn(
                        "Accuracy (%)",
                        format="%.1f%%"
                    )
            }
        )

        st.caption(
            "Model performance only includes "
            "fixtures for which each model "
            "was selected by Q-Learning."
        )
    st.divider()

    st.caption(
        "Original predictions are preserved "
        "before kickoff. Completed outcomes "
        "are registered during scheduled "
        "prediction refreshes."
    )



    

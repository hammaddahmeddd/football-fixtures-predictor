
import pandas as pd
import streamlit as st

from config import (
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
)


METRICS_DIR = OUTPUTS_DIR / "metrics"
GRAPH_DIR = OUTPUTS_DIR / "graphs"


@st.cache_data
def read_optional_csv(path):

    if not path.exists():
        return None

    return pd.read_csv(path)


def render():

    st.title("⚽ Goal Analysis")

    st.write(
        "Regression-based total-goal predictions "
        "and Over/Under 2.5 analysis."
    )

    results = read_optional_csv(
        METRICS_DIR
        / "regression_results.csv"
    )

    weekly = read_optional_csv(
        PREDICTIONS_DIR
        / "weekly_predictions.csv"
    )

    test_predictions = read_optional_csv(
        METRICS_DIR
        / "goal_test_predictions.csv"
    )

    st.subheader("Regression Evaluation")

    if results is not None:

        st.dataframe(
            results,
            hide_index=True,
            width="stretch"
        )

    else:

        st.warning(
            "Regression evaluation file is missing."
        )

    st.caption(
        "Lower MAE and RMSE are better. "
        "Negative R² means the regression model "
        "performed worse than predicting the "
        "evaluation target's mean."
    )

    st.divider()

    st.subheader("Actual vs Predicted Goals")

    actual_chart = (
        GRAPH_DIR
        / "regression_actual_vs_predicted.png"
    )

    if actual_chart.exists():

        st.image(
            str(actual_chart),
            width="stretch"
        )

    elif test_predictions is not None:

        st.dataframe(
            test_predictions.head(30),
            hide_index=True,
            width="stretch"
        )

    else:

        st.info(
            "Actual-vs-predicted chart not available."
        )

    st.subheader("Prediction Error Distribution")

    error_chart = (
        GRAPH_DIR
        / "regression_error_distribution.png"
    )

    if error_chart.exists():

        st.image(
            str(error_chart),
            width="stretch"
        )

    st.divider()

    st.subheader("Upcoming Fixture Goal Forecasts")

    if weekly is not None:

        required = [
            "HomeTeam",
            "AwayTeam",
            "PredictedTotalGoals",
            "OverUnder2_5",
        ]

        if all(
            col in weekly.columns
            for col in required
        ):

            st.dataframe(
                weekly[required],
                hide_index=True,
                width="stretch"
            )

            counts = (
                weekly["OverUnder2_5"]
                .value_counts()
                .rename_axis("Prediction")
                .reset_index(name="Matches")
            )

            st.bar_chart(
                counts.set_index("Prediction"),
                y_label="Number of Matches",
            )

    st.divider()

    st.warning(
        "The saved regression evaluation showed "
        "that the model did not outperform the "
        "simple mean-goals baseline on the test "
        "season. These forecasts should therefore "
        "be interpreted cautiously."
    )


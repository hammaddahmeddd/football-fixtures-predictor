# V4 CLEAN DASHBOARD ADAPTER
from dashboard import v4_data as v4

import pandas as pd
import streamlit as st

from config import OUTPUTS_DIR


METRICS_DIR = OUTPUTS_DIR / "metrics"
GRAPH_DIR = OUTPUTS_DIR / "graphs"


def load_season():
    return v4.season()


def render():

    st.title("🏆 Projected Premier League Table")

    st.write(
        "Full-season projections based on "
        "V4 Monte Carlo simulations (default 5,000)."
    )

    df = load_season()

    if df is None or df.empty:

        st.warning(
            "Season simulation results not found. "
            "Run python -m src.refresh_v4."
        )
        return

    display = df.copy()

    probability_columns = [
        "TitleProbability",
        "Top4Probability",
        "FifthPlaceProbability",
        "Top6Probability",
        "RelegationProbability",
    ]

    for column in probability_columns:

        if column in display.columns:

            display[column] = (
                display[column] * 100
            )

    st.subheader("Projected Final Standings")

    table_columns = [
        "ProjectedPosition",
        "Team",
        "ExpectedPoints",
        "ExpectedWins",
        "ExpectedDraws",
        "ExpectedLosses",
        "ExpectedGoalDifference",
    ]

    available = [
        col for col in table_columns
        if col in display.columns
    ]

    st.dataframe(
        display[available],
        hide_index=True,
        width="stretch",
    )

    st.divider()

    st.subheader("Season Outcome Probabilities")

    probability_view = [
        "ProjectedPosition",
        "Team",
        "TitleProbability",
        "Top4Probability",
        "FifthPlaceProbability",
        "RelegationProbability",
    ]

    available_probs = [
        col for col in probability_view
        if col in display.columns
    ]

    st.dataframe(
        display[available_probs],
        hide_index=True,
        width="stretch",
        column_config={
            col: st.column_config.NumberColumn(
                col.replace("Probability", " Chance"),
                format="%.1f%%"
            )
            for col in probability_columns
            if col in available_probs
        }
    )

    st.download_button(
        "Download Season Projections",
        df.to_csv(index=False),
        file_name="season_simulation_v4.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("League Outcome Charts")

    for col in ["ExpectedPoints", "Top4Probability", "RelegationProbability"]:
        st.bar_chart(display.set_index("Team")[[col]])

    st.divider()

    st.subheader("Team Projection Explorer")

    team = st.selectbox(
        "Choose a team",
        df["Team"].tolist(),
    )

    selected = df[
        df["Team"] == team
    ].iloc[0]

    a, b, c, d = st.columns(4)

    a.metric(
        "Expected Points",
        f"{selected['ExpectedPoints']:.1f}"
    )

    b.metric(
        "Title Chance",
        f"{selected['TitleProbability']:.1%}"
    )

    c.metric(
        "Top-Four Chance",
        f"{selected['Top4Probability']:.1%}"
    )

    d.metric(
        "Relegation Chance",
        f"{selected['RelegationProbability']:.1%}"
    )

    st.write(
        "Expected Final Position:",
        f"{selected['AveragePosition']:.1f}"
    )

    st.write(
        "90% Simulation Points Interval:",
        f"{selected['PointsP05']:.0f} to "
        f"{selected['PointsP95']:.0f}"
    )

    st.caption(
        "The probabilities come from simulated "
        "seasons. They depend on model assumptions "
        "and are not official league forecasts."
    )


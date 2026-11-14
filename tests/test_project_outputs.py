
"""
Automated tests for the Football Fixtures
Result Predictor.

Run:
    python -m pytest tests/test_project_outputs.py -v
"""

import pandas as pd
import pytest

from config import (
    RAW_DATA_DIR,
    PREDICTIONS_DIR,
    OUTPUTS_DIR,
)


def load_required_csv(path):
    assert path.exists(), f"Missing file: {path}"
    return pd.read_csv(path)


def test_historical_dataset():

    df = load_required_csv(
        RAW_DATA_DIR / "premier_league_raw.csv"
    )

    assert len(df) == 1900

    required = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]

    assert set(required).issubset(df.columns)
    assert df[required].notna().all().all()


def test_current_season_dataset():

    df = load_required_csv(
        RAW_DATA_DIR
        / "premier_league_2026_27.csv"
    )

    assert not df.empty

    assert (
        df["HomeTeam"] != df["AwayTeam"]
    ).all()

    assert df["FTHG"].ge(0).all()
    assert df["FTAG"].ge(0).all()


def test_weekly_predictions():

    df = load_required_csv(
        PREDICTIONS_DIR
        / "weekly_predictions.csv"
    )

    assert not df.empty
    assert len(df) <= 10

    assert df["PredictedResult"].isin(
        ["H", "D", "A"]
    ).all()

    assert df["SelectedModel"].isin(
        [
            "KNN",
            "SVM",
            "Decision Tree",
            "Random Forest",
        ]
    ).all()

    assert df["SelectedConfidence"].between(
        0, 1
    ).all()

    assert df["PredictedTotalGoals"].ge(
        0
    ).all()


def test_weekly_goal_labels():

    df = load_required_csv(
        PREDICTIONS_DIR
        / "weekly_predictions.csv"
    )

    expected = df[
        "PredictedTotalGoals"
    ].apply(
        lambda x: "Over" if x > 2.5 else "Under"
    )

    assert (
        df["OverUnder2_5"] == expected
    ).all()


def test_season_simulation():

    df = load_required_csv(
        OUTPUTS_DIR
        / "metrics"
        / "season_simulation_summary.csv"
    )

    assert len(df) == 20

    assert df["Team"].nunique() == 20

    assert sorted(
        df["ProjectedPosition"].tolist()
    ) == list(range(1, 21))

    assert df["ExpectedPoints"].ge(
        0
    ).all()


def test_season_probabilities():

    df = load_required_csv(
        OUTPUTS_DIR
        / "metrics"
        / "season_simulation_summary.csv"
    )

    columns = [
        "TitleProbability",
        "Top4Probability",
        "RelegationProbability",
    ]

    for column in columns:

        assert df[column].between(
            0, 1
        ).all()

    # Every simulated season has one champion,
    # four top-four teams and three relegated teams.
    assert df["TitleProbability"].sum() == pytest.approx(
        1.0,
        abs=0.00001
    )

    assert df["Top4Probability"].sum() == pytest.approx(
        4.0,
        abs=0.00001
    )

    assert df["RelegationProbability"].sum() == pytest.approx(
        3.0,
        abs=0.00001
    )


def test_remaining_fixtures():

    completed = load_required_csv(
        RAW_DATA_DIR
        / "premier_league_2026_27.csv"
    )

    remaining = load_required_csv(
        PREDICTIONS_DIR
        / "remaining_fixtures_predictions.csv"
    )

    assert len(completed) + len(remaining) == 380

    assert remaining[
        ["HomeTeam", "AwayTeam"]
    ].notna().all().all()

    assert not remaining.duplicated(
        subset=["HomeTeam", "AwayTeam"]
    ).any()


def test_simulated_points():

    points = load_required_csv(
        OUTPUTS_DIR
        / "metrics"
        / "simulated_season_points.csv"
    )

    summary = load_required_csv(
        OUTPUTS_DIR
        / "metrics"
        / "season_simulation_summary.csv"
    )

    assert len(points) == 1000

    for _, team in summary.iterrows():

        name = team["Team"]

        assert name in points.columns

        assert points[name].mean() == pytest.approx(
            team["ExpectedPoints"],
            abs=0.01
        )

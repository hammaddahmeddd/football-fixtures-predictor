
"""
Football Fixtures Result Predictor V2
Automated Project Tests

Run:
    python -m pytest tests/test_project_outputs.py -v
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
)

from src.feature_engineering import FEATURE_COLUMNS


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def load_required_csv(path):

    assert path.exists(), f"Missing file: {path}"

    return pd.read_csv(path)


# =====================================================
# TEST 1: HISTORICAL DATASET
# =====================================================

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
        "Season",
    ]

    assert set(required).issubset(df.columns)

    assert df[required].notna().all().all()

    assert df["FTR"].isin(["H", "D", "A"]).all()

    assert df["FTHG"].ge(0).all()
    assert df["FTAG"].ge(0).all()

    assert not df.duplicated(
        subset=["Season", "HomeTeam", "AwayTeam"]
    ).any()


# =====================================================
# TEST 2: CURRENT SEASON DATASET
# =====================================================

def test_current_season_dataset():

    df = load_required_csv(
        RAW_DATA_DIR / "premier_league_2026_27.csv"
    )

    assert not df.empty

    assert len(df) <= 380

    assert (
        df["HomeTeam"] != df["AwayTeam"]
    ).all()

    assert df["FTHG"].ge(0).all()
    assert df["FTAG"].ge(0).all()

    assert df["FTR"].isin(["H", "D", "A"]).all()

    expected_results = np.where(
        df["FTHG"] > df["FTAG"],
        "H",
        np.where(
            df["FTHG"] < df["FTAG"],
            "A",
            "D"
        )
    )

    assert (
        df["FTR"].to_numpy() == expected_results
    ).all()

    assert not df.duplicated(
        subset=["HomeTeam", "AwayTeam"]
    ).any()


# =====================================================
# TEST 3: ADVANCED FEATURE DATASET
# =====================================================

def test_advanced_feature_dataset():

    df = load_required_csv(
        PROCESSED_DATA_DIR /
        "premier_league_features.csv"
    )

    assert len(df) == 1900

    assert len(FEATURE_COLUMNS) == 38

    assert set(FEATURE_COLUMNS).issubset(
        df.columns
    )

    assert df[FEATURE_COLUMNS].notna().all().all()

    assert np.isfinite(
        df[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all()

    required_features = [
        "HomeElo",
        "AwayElo",
        "EloDifference",
        "HomeWeightedPoints",
        "AwayWeightedPoints",
        "HomeLast10GoalsFor",
        "AwayLast10GoalsFor",
    ]

    assert set(required_features).issubset(
        df.columns
    )


# =====================================================
# TEST 4: TRAINED MODEL FEATURE COMPATIBILITY
# =====================================================

def test_model_feature_compatibility():

    feature_path = (
        MODELS_DIR / "feature_columns.joblib"
    )

    assert feature_path.exists()

    trained_features = joblib.load(feature_path)

    assert len(trained_features) == 38

    assert list(trained_features) == list(
        FEATURE_COLUMNS
    )

    model_files = [
        "knn.joblib",
        "svm.joblib",
        "decision_tree.joblib",
        "random_forest.joblib",
        "best_classifier.joblib",
        "goal_regressor.joblib",
        "over_under_classifier.joblib",
        "q_learning_policy.joblib",
    ]

    for filename in model_files:

        assert (MODELS_DIR / filename).exists(), (
            f"Missing trained model: {filename}"
        )


# =====================================================
# TEST 5: CLASSIFIER PREDICTION PROBABILITIES
# =====================================================

def test_classifier_probabilities():

    df = load_required_csv(
        PROCESSED_DATA_DIR /
        "premier_league_features.csv"
    )

    X = df[FEATURE_COLUMNS].tail(20)

    model_files = [
        "knn.joblib",
        "svm.joblib",
        "decision_tree.joblib",
        "random_forest.joblib",
    ]

    for filename in model_files:

        model = joblib.load(
            MODELS_DIR / filename
        )

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)

        assert len(predictions) == len(X)

        assert probabilities.shape == (
            len(X), 3
        )

        assert np.isfinite(probabilities).all()

        assert (probabilities >= 0).all()

        assert (probabilities <= 1).all()

        assert np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=0.00001
        )

        assert set(model.classes_) == {
            "H", "D", "A"
        }

        assert set(predictions).issubset(
            {"H", "D", "A"}
        )


# =====================================================
# TEST 6: WEEKLY MATCH PREDICTIONS
# =====================================================

def test_weekly_predictions():

    df = load_required_csv(
        PREDICTIONS_DIR /
        "weekly_predictions.csv"
    )

    assert not df.empty

    assert len(df) <= 10

    required = [
        "MatchID",
        "HomeTeam",
        "AwayTeam",
        "PredictedResult",
        "SelectedModel",
        "SelectedConfidence",
        "PredictedTotalGoals",
        "OverUnder2_5",
        "ProbOver2_5",
    ]

    assert set(required).issubset(df.columns)

    assert df[required].notna().all().all()

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

    assert df["MatchID"].is_unique

    assert (
        df["HomeTeam"] != df["AwayTeam"]
    ).all()


# =====================================================
# TEST 7: OVER / UNDER 2.5 GOAL PREDICTIONS
# =====================================================

def test_weekly_goal_labels():

    df = load_required_csv(
        PREDICTIONS_DIR /
        "weekly_predictions.csv"
    )

    assert "ProbOver2_5" in df.columns

    assert df["ProbOver2_5"].notna().all()

    assert df["ProbOver2_5"].between(
        0, 1
    ).all()

    assert df["OverUnder2_5"].isin(
        ["Over", "Under"]
    ).all()

    # Over/Under is now determined by probability,
    # not by the predicted average total goals.

    expected = np.where(
        df["ProbOver2_5"] >= 0.5,
        "Over",
        "Under"
    )

    assert (
        df["OverUnder2_5"].to_numpy() == expected
    ).all()

    # Validate Under probability if it is exported.

    if "ProbUnder2_5" in df.columns:

        assert df["ProbUnder2_5"].between(
            0, 1
        ).all()

        assert np.allclose(
            df["ProbOver2_5"]
            + df["ProbUnder2_5"],
            1.0,
            atol=0.00001
        )


# =====================================================
# TEST 8: Q-LEARNING POLICY
# =====================================================

def test_q_learning_policy():

    policy = joblib.load(
        MODELS_DIR / "q_learning_policy.joblib"
    )

    required = [
        "q_table",
        "actions",
        "gamma",
        "learning_rate",
        "fallback_model",
    ]

    assert set(required).issubset(policy.keys())

    assert len(policy["q_table"]) > 0

    assert len(policy["actions"]) == 4

    assert policy["gamma"] == 0

    assert policy["fallback_model"] in (
        policy["actions"]
    )

    for state, values in policy["q_table"].items():

        values = np.asarray(
            values,
            dtype=float
        )

        assert len(values) == 4

        assert np.isfinite(values).all()

        assert (values >= 0).all()

        assert (values <= 1).all()


# =====================================================
# TEST 9: SEASON SIMULATION
# =====================================================

def test_season_simulation():

    df = load_required_csv(
        OUTPUTS_DIR /
        "metrics" /
        "season_simulation_summary.csv"
    )

    assert len(df) == 20

    assert df["Team"].nunique() == 20

    assert sorted(
        df["ProjectedPosition"].tolist()
    ) == list(range(1, 21))

    assert df["ExpectedPoints"].between(
        0, 114
    ).all()

    assert df["AveragePosition"].between(
        1, 20
    ).all()


# =====================================================
# TEST 10: SEASON PROBABILITIES
# =====================================================

def test_season_probabilities():

    df = load_required_csv(
        OUTPUTS_DIR /
        "metrics" /
        "season_simulation_summary.csv"
    )

    columns = [
        "TitleProbability",
        "Top4Probability",
        "RelegationProbability",
    ]

    for column in columns:

        assert df[column].notna().all()

        assert df[column].between(
            0, 1
        ).all()

    # Every season has one champion.

    assert df["TitleProbability"].sum() == (
        pytest.approx(1.0, abs=0.00001)
    )

    # Every season has four top-four teams.

    assert df["Top4Probability"].sum() == (
        pytest.approx(4.0, abs=0.00001)
    )

    # Every season has three relegated teams.

    assert df["RelegationProbability"].sum() == (
        pytest.approx(3.0, abs=0.00001)
    )


# =====================================================
# TEST 11: REMAINING FIXTURES
# =====================================================

def test_remaining_fixtures():

    completed = load_required_csv(
        RAW_DATA_DIR /
        "premier_league_2026_27.csv"
    )

    remaining = load_required_csv(
        PREDICTIONS_DIR /
        "remaining_fixtures_predictions.csv"
    )

    assert len(completed) + len(remaining) == 380

    assert remaining[
        ["HomeTeam", "AwayTeam"]
    ].notna().all().all()

    assert not remaining.duplicated(
        subset=["HomeTeam", "AwayTeam"]
    ).any()

    completed_pairs = set(
        zip(
            completed["HomeTeam"],
            completed["AwayTeam"]
        )
    )

    remaining_pairs = set(
        zip(
            remaining["HomeTeam"],
            remaining["AwayTeam"]
        )
    )

    assert completed_pairs.isdisjoint(
        remaining_pairs
    )

    season = pd.concat(
        [completed, remaining],
        ignore_index=True
    )

    appearances = pd.concat(
        [
            season["HomeTeam"],
            season["AwayTeam"],
        ]
    ).value_counts()

    assert len(appearances) == 20

    assert (appearances == 38).all()


# =====================================================
# TEST 12: SIMULATED MATCH PROBABILITIES
# =====================================================

def test_simulated_match_probabilities():

    df = load_required_csv(
        PREDICTIONS_DIR /
        "remaining_fixtures_predictions.csv"
    )

    columns = [
        "ProbHome",
        "ProbDraw",
        "ProbAway",
    ]

    assert set(columns).issubset(df.columns)

    probabilities = df[columns].to_numpy(
        dtype=float
    )

    assert np.isfinite(probabilities).all()

    assert (probabilities >= 0).all()

    assert (probabilities <= 1).all()

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=0.00001
    )


# =====================================================
# TEST 13: MONTE CARLO SIMULATED POINTS
# =====================================================

def test_simulated_points():

    points = load_required_csv(
        OUTPUTS_DIR /
        "metrics" /
        "simulated_season_points.csv"
    )

    summary = load_required_csv(
        OUTPUTS_DIR /
        "metrics" /
        "season_simulation_summary.csv"
    )

    assert len(points) == 1000

    for _, team in summary.iterrows():

        name = team["Team"]

        assert name in points.columns

        assert points[name].between(
            0, 114
        ).all()

        assert points[name].mean() == (
            pytest.approx(
                team["ExpectedPoints"],
                abs=0.01
            )
        )

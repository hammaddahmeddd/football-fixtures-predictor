
"""
Premier League total goals regression.

Predicts the total number of goals in a match.

Training:   2021-22 to 2023-24
Validation: 2024-25
Testing:    2025-26
"""

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)

from src.feature_engineering import FEATURE_COLUMNS


TRAIN_SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
]

VALIDATION_SEASON = "2024-25"
TEST_SEASON = "2025-26"


def load_regression_data():
    """Load the feature-engineered dataset."""

    file_path = (
        PROCESSED_DATA_DIR
        / "premier_league_features.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "Feature dataset not found. Run "
            "python -m src.feature_engineering first."
        )

    df = pd.read_csv(file_path)

    required = FEATURE_COLUMNS + [
        "Date",
        "Season",
        "HomeTeam",
        "AwayTeam",
        "TotalGoals",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[required].isnull().any().any():
        raise ValueError(
            "Missing values found in regression data."
        )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    print(f"Loaded {len(df)} matches.")

    return df


def split_regression_data(df):
    """Split football data by season."""

    train = df[
        df["Season"].isin(TRAIN_SEASONS)
    ].copy()

    validation = df[
        df["Season"] == VALIDATION_SEASON
    ].copy()

    test = df[
        df["Season"] == TEST_SEASON
    ].copy()

    if any(
        len(part) == 0
        for part in [train, validation, test]
    ):
        raise ValueError(
            "One or more regression splits are empty."
        )

    X_train = train[FEATURE_COLUMNS]
    y_train = train["TotalGoals"]

    X_val = validation[FEATURE_COLUMNS]
    y_val = validation["TotalGoals"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["TotalGoals"]

    print("\n===== DATA SPLIT =====")
    print("Training matches:", len(train))
    print("Validation matches:", len(validation))
    print("Testing matches:", len(test))

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        test,
    )


def create_regression_model():
    """Create Random Forest regression model."""

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )

    return model


def evaluate_regression(y_true, y_pred, name):
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(
        y_true, y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(
        y_true, y_pred
    )

    results = {
        "Model": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    return results


def save_regression_graphs(y_true, y_pred):
    """Create plots for the project report."""

    graph_dir = OUTPUTS_DIR / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        actual,
        predicted,
        alpha=0.5
    )

    max_goals = max(
        float(actual.max()),
        float(predicted.max())
    )

    ax.plot(
        [0, max_goals],
        [0, max_goals],
        "r--",
        label="Perfect prediction"
    )

    ax.set_xlabel("Actual Total Goals")
    ax.set_ylabel("Predicted Total Goals")
    ax.set_title("Actual vs Predicted Goals")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        graph_dir / "regression_actual_vs_predicted.png",
        dpi=150
    )

    plt.close(fig)

    # Prediction errors
    errors = actual - predicted

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        errors,
        bins=25,
        edgecolor="black"
    )

    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Goal Prediction Error Distribution")

    fig.tight_layout()

    fig.savefig(
        graph_dir / "regression_error_distribution.png",
        dpi=150
    )

    plt.close(fig)

    print("\nRegression graphs saved.")


def train_goal_regressor():
    """Train, evaluate and save regression model."""

    df = load_regression_data()

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        test_df,
    ) = split_regression_data(df)

    print("\n===== TRAINING REGRESSION MODEL =====")

    model = create_regression_model()

    model.fit(X_train, y_train)

    # Evaluate validation performance
    val_predictions = model.predict(X_val)

    validation_metrics = evaluate_regression(
        y_val,
        val_predictions,
        "Random Forest Validation"
    )

    # Final held-out test
    test_predictions = model.predict(X_test)

    test_metrics = evaluate_regression(
        y_test,
        test_predictions,
        "Random Forest Test"
    )

    # Simple baseline: predict training mean
    baseline_predictions = np.full(
        len(y_test),
        float(y_train.mean())
    )

    baseline_metrics = evaluate_regression(
        y_test,
        baseline_predictions,
        "Training Mean Baseline Test"
    )

    # Evaluate Over/Under threshold
    predicted_over = (
        test_predictions > 2.5
    ).astype(int)

    actual_over = (
        y_test.to_numpy() > 2.5
    ).astype(int)

    over_under_accuracy = (
        predicted_over == actual_over
    ).mean()

    print("\n===== OVER/UNDER 2.5 =====")
    print(
        f"Test accuracy: "
        f"{over_under_accuracy:.4f}"
    )

    # Save model
    MODELS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(
        model,
        MODELS_DIR / "goal_regressor.joblib"
    )

    # Save metrics
    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(
        parents=True, exist_ok=True
    )

    results_df = pd.DataFrame([
        {
            "Split": "Validation",
            **validation_metrics
        },
        {
            "Split": "Test",
            **test_metrics
        },
        {
            "Split": "Test Baseline",
            **baseline_metrics
        },
    ])

    results_df.to_csv(
        metrics_dir / "regression_results.csv",
        index=False
    )

    # Save individual test predictions
    prediction_df = test_df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "TotalGoals",
        ]
    ].copy()

    prediction_df["PredictedGoals"] = (
        test_predictions
    )

    prediction_df["ActualOver2_5"] = (
        actual_over
    )

    prediction_df["PredictedOver2_5"] = (
        predicted_over
    )

    prediction_df.to_csv(
        metrics_dir / "goal_test_predictions.csv",
        index=False
    )

    save_regression_graphs(
        y_test,
        test_predictions
    )

    print("\n===== COMPLETED =====")
    print("Goal regression model saved.")
    print("Regression metrics saved.")
    print("Test predictions saved.")

    return model


if __name__ == "__main__":
    train_goal_regressor()

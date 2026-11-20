"""Select total-goal regressor and independent O/U 2.5 classifier on validation season.

Run: python -m src.regression.goal_regressor
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, log_loss, brier_score_loss
from config import PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from src.feature_engineering import FEATURE_COLUMNS


def main():
    df = pd.read_csv(PROCESSED_DATA_DIR / "premier_league_features.csv")
    if not set(FEATURE_COLUMNS).issubset(df.columns):
        raise ValueError("Missing enhanced features. Rerun src.feature_engineering")
    train = df[df.Season.isin(["2021-22", "2022-23", "2023-24"])]
    val = df[df.Season == "2024-25"]
    test = df[df.Season == "2025-26"]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Missing split data")
    xtr, xv, xt = [d[FEATURE_COLUMNS] for d in (train, val, test)]
    ytr, yv, yt = [d["TotalGoals"].to_numpy(dtype=float) for d in (train, val, test)]
    MODEL = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=100)),
        "Poisson": make_pipeline(StandardScaler(), PoissonRegressor(alpha=3.0, max_iter=1500)),
        "RandomForest": RandomForestRegressor(n_estimators=220, max_depth=5, min_samples_leaf=12, random_state=42, n_jobs=-1),
        "GradientBoosting": HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=9, l2_regularization=15, learning_rate=0.04, random_state=42),
    }
    # A train-era mean predictor is a legitimate competing candidate.
    baseline = float(ytr.mean())
    best_name = "TrainMean"
    best_mae = mean_absolute_error(yv, np.full(len(yv), baseline))
    best_model = None
    val_rows = [{"Model": "TrainMean", "ValidationMAE": best_mae}]
    for name, model in MODEL.items():
        model.fit(xtr, ytr)
        estimate = np.maximum(0, model.predict(xv))
        score = mean_absolute_error(yv, estimate)
        val_rows.append({"Model": name, "ValidationMAE": score})
        if score < best_mae:
            best_name, best_mae, best_model = name, score, model
    if best_model is None:
        from sklearn.dummy import DummyRegressor
        best_model = DummyRegressor(strategy="mean").fit(xtr, ytr)
    train_binary = (ytr > 2.5).astype(int)
    val_binary = (yv > 2.5).astype(int)
    test_binary = (yt > 2.5).astype(int)
    over_candidates = {
        "TrainPrior": DummyClassifier(strategy="prior"),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1200, random_state=42)),
        "GradientBoosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=8, l2_regularization=15, learning_rate=0.04, random_state=42),
    }
    best_over_model, best_over_name, best_over_loss = None, None, float("inf")
    over_rows = []
    for name, model in over_candidates.items():
        model.fit(xtr, train_binary)
        pr = np.clip(model.predict_proba(xv)[:, 1], 1e-6, 1-1e-6)
        loss = log_loss(val_binary, pr, labels=[0, 1])
        over_rows.append({"Model": name, "ValidationLogLoss": loss,
                          "ValidationAccuracy": accuracy_score(val_binary, (pr >= 0.5).astype(int))})
        if loss < best_over_loss:
            best_over_name, best_over_model, best_over_loss = name, model, loss

    pred_test = np.maximum(0, best_model.predict(xt))
    over_prob = best_over_model.predict_proba(xt)[:, 1]
    over_pred = (over_prob >= 0.5).astype(int)
    baseline_test = np.full(len(yt), baseline)
    metrics = []
    score_models = [(best_name, pred_test)]
    if best_name != "TrainMean":
        score_models.append(("TrainMean", baseline_test))
    for name, predictions in score_models:
        metrics.append({"Model": name, "Split": "Test", "MAE": mean_absolute_error(yt, predictions),
                        "RMSE": np.sqrt(mean_squared_error(yt, predictions)), "R2": r2_score(yt, predictions),
                        "OverUnderAccuracy": accuracy_score(test_binary, (predictions > 2.5).astype(int))})
    metrics.append({"Model": f"OverUnder:{best_over_name}", "Split": "Test", "MAE": np.nan, "RMSE": np.nan,
                    "R2": np.nan, "OverUnderAccuracy": accuracy_score(test_binary, over_pred),
                    "BinaryLogLoss": log_loss(test_binary, over_prob, labels=[0, 1]),
                    "BinaryBrier": brier_score_loss(test_binary, over_prob)})
    prior_prob = np.full(len(test_binary), train_binary.mean())
    metrics.append({"Model": "OverUnder:TrainPriorBaseline", "Split": "Test", "MAE": np.nan,
                    "RMSE": np.nan, "R2": np.nan,
                    "OverUnderAccuracy": accuracy_score(test_binary, (prior_prob >= 0.5).astype(int)),
                    "BinaryLogLoss": log_loss(test_binary, prior_prob, labels=[0, 1]),
                    "BinaryBrier": brier_score_loss(test_binary, prior_prob)})
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    md = OUTPUTS_DIR / "metrics"
    gd = OUTPUTS_DIR / "graphs"
    md.mkdir(parents=True, exist_ok=True)
    gd.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "goal_regressor.joblib")
    joblib.dump(best_over_model, MODELS_DIR / "over_under_classifier.joblib")
    pd.DataFrame(val_rows).to_csv(md / "regression_validation_comparison.csv", index=False)
    pd.DataFrame(over_rows).to_csv(md / "over_under_validation_comparison.csv", index=False)
    pd.DataFrame(metrics).to_csv(md / "regression_results.csv", index=False)
    pd.DataFrame({"ActualTotalGoals": yt, "PredictedTotalGoals": pred_test,
                  "ActualOver2_5": test_binary, "ProbOver2_5": over_prob, "PredictedOver2_5": over_pred}).to_csv(
                  md / "goal_test_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(yt, pred_test, alpha=0.35)
    ax.plot([0, max(yt.max(), pred_test.max())], [0, max(yt.max(), pred_test.max())], ls="--")
    ax.set(xlabel="Actual total goals", ylabel="Predicted total goals", title="Goal Prediction vs Actual")
    fig.tight_layout()
    fig.savefig(gd / "regression_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(yt - pred_test, bins=22)
    ax.set(xlabel="Actual minus predicted goals", ylabel="Matches", title="Prediction Error Distribution")
    fig.tight_layout()
    fig.savefig(gd / "regression_error_distribution.png", dpi=150)
    plt.close(fig)
    print("Validation-selected goal model:", best_name)
    print("Validation-selected Over/Under model:", best_over_name)
    print(pd.DataFrame(metrics).to_string(index=False))
    print("Models saved. Evaluate test once; do not tune to this final test.")


if __name__ == "__main__":
    main()

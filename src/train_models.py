"""Time-aware hyperparameter search, isolated validation selection, historical test comparison.

Run after feature_engineering: python -m src.train_models
Preserves existing CSV and model names used by Streamlit and Q-Learning.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

from config import PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from src.feature_engineering import FEATURE_COLUMNS
from src.evaluate_models import evaluate_classifier, save_confusion_matrix, save_results_table

SPLITS = {
    "train": ["2021-22", "2022-23", "2023-24"],
    "validation": ["2024-25"],
    "test": ["2025-26"],
}


def load_training_data():
    df = pd.read_csv(PROCESSED_DATA_DIR / "premier_league_features.csv")
    required = FEATURE_COLUMNS + ["Season", "FTR"]
    if not set(required).issubset(df.columns):
        raise ValueError("Feature dataset missing advanced columns; rerun src.feature_engineering")
    if df[required].isna().any().any() or not df["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError("Invalid features or outcome labels")
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def split_data(df):
    portions = [df[df.Season.isin(SPLITS[k])] for k in ("train", "validation", "test")]
    if any(p.empty for p in portions):
        raise ValueError("Training, validation, and test splits must all be populated")
    result = []
    for p in portions:
        result.extend((p[FEATURE_COLUMNS], p["FTR"]))
    return tuple(result)


def model_candidates():
    return {
        "KNN": [make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n, weights=w))
                for n in [11, 21, 35] for w in ["uniform", "distance"]],
        "SVM": [make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", gamma="scale", random_state=42))
                for c in [0.3, 1.0, 3.0]],
        "Decision Tree": [DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf, random_state=42)
                          for d in [3, 5, 7] for leaf in [10, 20]],
        "Random Forest": [RandomForestClassifier(n_estimators=220, max_depth=d,
                           min_samples_leaf=leaf, max_features=0.8, random_state=42, n_jobs=-1)
                          for d in [5, 9] for leaf in [5, 12]],
    }


def train_all_models():
    df = load_training_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    # Internal chronological tuning: never use 2024/25 validation or 2025/26 test to tune candidates.
    internal_train = df[df.Season.isin(["2021-22", "2022-23"])]
    internal_val = df[df.Season == "2023-24"]
    if internal_train.empty or internal_val.empty:
        raise ValueError("Not enough data for chronological tuning")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metric_dir = OUTPUTS_DIR / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    tuning_rows = []
    trained = {}
    val_results = []

    for name, candidates in model_candidates().items():
        winner = None
        best_score = -np.inf
        for index, candidate in enumerate(candidates):
            candidate.fit(internal_train[FEATURE_COLUMNS], internal_train.FTR)
            labels = candidate.predict(internal_val[FEATURE_COLUMNS])
            score = f1_score(internal_val.FTR, labels, labels=["H", "D", "A"], average="macro", zero_division=0)
            tuning_rows.append({"Model": name, "Candidate": index, "InternalMacroF1": score,
                                "Parameters": str(candidate.get_params(deep=False))})
            if score > best_score:
                best_score, winner = score, candidate
        # Refit only using all three training seasons.
        winner.fit(X_train, y_train)
        # Calibrate only on time-ordered training folds. Compare on 2024/25
        # validation; never inspect the 2025/26 test to choose calibration.
        calibration = CalibratedClassifierCV(
            estimator=clone(winner), method="sigmoid",
            cv=TimeSeriesSplit(n_splits=3), ensemble=True,
        )
        calibration.fit(X_train, y_train)
        cal_loss = log_loss(y_val, calibration.predict_proba(X_val), labels=calibration.classes_)
        if name == "SVM":
            winner = calibration
            details = f"cal_val_logloss={cal_loss:.4f};selected=calibrated"
        else:
            raw_loss = log_loss(y_val, winner.predict_proba(X_val), labels=winner.classes_)
            if cal_loss < raw_loss:
                winner = calibration
            details = f"raw_val_logloss={raw_loss:.4f};cal_val_logloss={cal_loss:.4f};selected={'calibrated' if cal_loss < raw_loss else 'raw'}"
        tuning_rows.append({"Model": name, "Candidate": "calibration", "InternalMacroF1": np.nan,
                            "Parameters": details})
        trained[name] = winner
        val_results.append(evaluate_classifier(winner, X_val, y_val, name, "Validation"))
        safe = name.lower().replace(" ", "_")
        joblib.dump(winner, MODELS_DIR / f"{safe}.joblib")
        print(f"{name}: internal F1={best_score:.4f}")

    pd.DataFrame(tuning_rows).to_csv(metric_dir / "classification_tuning.csv", index=False)
    val_results.sort(key=lambda r: (r["F1"], r["Accuracy"]), reverse=True)
    best_name = val_results[0]["Model"]
    joblib.dump(trained[best_name], MODELS_DIR / "best_classifier.joblib")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.joblib")

    test_results = []
    for name, model in trained.items():
        test_results.append(evaluate_classifier(model, X_test, y_test, name, "Test"))
        save_confusion_matrix(model, X_test, y_test, name)
    result = save_results_table(val_results + test_results)
    print("\nValidation winner:", best_name)
    print(result.to_string(index=False))
    print("Test results are for final evaluation only. Do not select candidates from them; this season has already been evaluated for v1.")
    return trained[best_name]


if __name__ == "__main__":
    train_all_models()

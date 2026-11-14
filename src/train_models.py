
import joblib
import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
)

from src.feature_engineering import FEATURE_COLUMNS

from src.classification.knn import create_knn_model
from src.classification.svm import create_svm_model
from src.classification.decision_tree import (
    create_decision_tree_model,
)
from src.ensemble.random_forest import (
    create_random_forest_model,
)

from src.evaluate_models import (
    evaluate_classifier,
    save_confusion_matrix,
    save_results_table,
)


def load_training_data():
    path = (
        PROCESSED_DATA_DIR /
        "premier_league_features.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Feature dataset not found. Run "
            "python -m src.feature_engineering first."
        )

    df = pd.read_csv(path)

    required = FEATURE_COLUMNS + ["FTR", "Season"]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError(
            "Missing values detected in training data."
        )

    if not df["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError("Unexpected result labels.")

    return df


def split_data(df):
    training_seasons = [
        "2021-22", "2022-23", "2023-24"
    ]

    validation_seasons = ["2024-25"]
    test_seasons = ["2025-26"]

    train = df[
        df["Season"].isin(training_seasons)
    ].copy()

    validation = df[
        df["Season"].isin(validation_seasons)
    ].copy()

    test = df[
        df["Season"].isin(test_seasons)
    ].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "One or more dataset splits are empty."
        )

    X_train = train[FEATURE_COLUMNS]
    y_train = train["FTR"]

    X_val = validation[FEATURE_COLUMNS]
    y_val = validation["FTR"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["FTR"]

    print("Training matches:", len(train))
    print("Validation matches:", len(validation))
    print("Testing matches:", len(test))

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )


def train_all_models():
    df = load_training_data()

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    ) = split_data(df)

    models = {
        "KNN": create_knn_model(),
        "SVM": create_svm_model(),
        "Decision Tree": create_decision_tree_model(),
        "Random Forest": create_random_forest_model(),
    }

    validation_results = []
    trained_models = {}

    print("\n===== TRAINING MODELS =====")

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_classifier(
            model, X_val, y_val, name, "Validation"
        )

        validation_results.append(metrics)

        print(
            f"Accuracy: {metrics['Accuracy']:.4f}"
        )
        print(
            f"Macro F1: {metrics['F1']:.4f}"
        )

        safe_name = name.lower().replace(" ", "_")

        joblib.dump(
            model,
            MODELS_DIR / f"{safe_name}.joblib"
        )

    # Select winner using validation F1 only
    validation_results.sort(
        key=lambda item: (
            item["F1"],
            item["Accuracy"]
        ),
        reverse=True
    )

    best_name = validation_results[0]["Model"]

    print(f"\nBest validation model: {best_name}")

    # Evaluate all trained models on the held-out
    # test season, without using its scores to
    # change the selected model.
    test_results = []

    for name, model in trained_models.items():
        metrics = evaluate_classifier(
            model, X_test, y_test, name, "Test"
        )

        test_results.append(metrics)

        save_confusion_matrix(
            model, X_test, y_test, name
        )

    # Save original models and validation-selected winner
    best_model = trained_models[best_name]

    joblib.dump(
        best_model,
        MODELS_DIR / "best_classifier.joblib"
    )

    joblib.dump(
        FEATURE_COLUMNS,
        MODELS_DIR / "feature_columns.joblib"
    )

    results_df = save_results_table(
        validation_results + test_results
    )

    print("\n===== MODEL COMPARISON =====")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nFinal selected model:", best_name)
    print("Model files saved to:", MODELS_DIR)

    return best_model


if __name__ == "__main__":
    train_all_models()

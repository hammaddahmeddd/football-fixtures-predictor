
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from config import OUTPUTS_DIR


LABELS = ["H", "D", "A"]
DISPLAY_LABELS = ["Home Win", "Draw", "Away Win"]


def evaluate_classifier(model, X, y, name, split):
    predictions = model.predict(X)

    results = {
        "Model": name,
        "Split": split,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y, predictions,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
        "F1": f1_score(
            y, predictions,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
    }

    return results


def save_confusion_matrix(model, X, y, name):
    graph_dir = OUTPUTS_DIR / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    predictions = model.predict(X)

    cm = confusion_matrix(
        y, predictions, labels=LABELS
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=DISPLAY_LABELS
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(f"{name} Confusion Matrix")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    path = graph_dir / f"{safe_name}_confusion_matrix.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved confusion matrix: {path}")


def save_results_table(results):
    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    path = metrics_dir / "classification_results.csv"
    df.to_csv(path, index=False)

    return df

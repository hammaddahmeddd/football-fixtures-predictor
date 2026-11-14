
"""
Support Vector Machine classifier.

Predicts:
H = Home Win
D = Draw
A = Away Win

Uses calibrated probabilities and feature scaling.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_svm_model():
    """
    Create a calibrated SVM classifier.

    StandardScaler is fitted inside each training
    fold to prevent data leakage.

    TimeSeriesSplit keeps calibration training
    data earlier than its validation data.

    The caller must provide chronologically
    ordered training matches.
    """

    svm_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight=None,
                random_state=42
            )
        )
    ])

    calibration_cv = TimeSeriesSplit(
        n_splits=3
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=svm_pipeline,
        method="sigmoid",
        cv=calibration_cv,
        ensemble=True
    )

    return calibrated_model

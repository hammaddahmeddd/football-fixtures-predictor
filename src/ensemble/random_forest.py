
from sklearn.ensemble import RandomForestClassifier


def create_random_forest_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

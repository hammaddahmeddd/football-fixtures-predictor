
from sklearn.tree import DecisionTreeClassifier


def create_decision_tree_model():
    return DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=15,
        random_state=42
    )

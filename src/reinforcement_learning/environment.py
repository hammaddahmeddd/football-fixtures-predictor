
"""
Football Prediction Strategy Environment.

An offline, one-step environment in which the
agent selects one of four classification models.

State:
    1. Do models disagree?
    2. Highest model confidence category
    3. Recent team-form difference category

Action:
    Choose KNN, SVM, Decision Tree, or Random Forest.

Reward:
    1 = selected model predicted correctly
    0 = selected model predicted incorrectly

Each fixture is a one-step episode.
"""

import pandas as pd


ACTIONS = [
    "KNN",
    "SVM",
    "Decision Tree",
    "Random Forest",
]

PREDICTION_COLUMNS = {
    "KNN": "KNN_Prediction",
    "SVM": "SVM_Prediction",
    "Decision Tree": "Decision_Tree_Prediction",
    "Random Forest": "Random_Forest_Prediction",
}

CONFIDENCE_COLUMNS = {
    "KNN": "KNN_Confidence",
    "SVM": "SVM_Confidence",
    "Decision Tree": "Decision_Tree_Confidence",
    "Random Forest": "Random_Forest_Confidence",
}


def create_state(row):
    """
    Create a discrete state from pre-match information.

    The actual result (FTR) is never used here.
    """

    predictions = [
        str(row[PREDICTION_COLUMNS[action]])
        for action in ACTIONS
    ]

    # 0 = agreement, 1 = disagreement
    disagreement = int(
        len(set(predictions)) > 1
    )

    highest_confidence = max(
        float(row[CONFIDENCE_COLUMNS[action]])
        for action in ACTIONS
    )

    if highest_confidence < 0.45:
        confidence_group = 0
    elif highest_confidence < 0.65:
        confidence_group = 1
    else:
        confidence_group = 2

    form_difference = float(
        row["FormDifference"]
    )

    if form_difference < -3:
        form_group = 0
    elif form_difference > 3:
        form_group = 2
    else:
        form_group = 1

    return (
        disagreement,
        confidence_group,
        form_group,
    )


class FootballStrategyEnvironment:
    """
    One-step Q-Learning environment.

    Each observation represents a completed
    historical football match.

    Its state only contains information that
    was available before the match.
    """

    def __init__(self, dataframe):

        self.data = dataframe.reset_index(
            drop=True
        ).copy()

        self.current_index = None

        required = (
            ["FTR", "FormDifference"]
            + list(PREDICTION_COLUMNS.values())
            + list(CONFIDENCE_COLUMNS.values())
        )

        missing = [
            col for col in required
            if col not in self.data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing environment columns: {missing}"
            )

        if self.data.empty:
            raise ValueError(
                "Environment has no match records."
            )

        if self.data[required].isnull().any().any():
            raise ValueError(
                "Missing values in environment data."
            )

    def reset(self, match_index):
        """
        Start an episode for one historical match.

        Returns a state without revealing the
        actual match outcome.
        """

        if not 0 <= match_index < len(self.data):
            raise IndexError(
                "Match index is out of range."
            )

        self.current_index = match_index

        row = self.data.iloc[match_index]

        return create_state(row)

    def step(self, action):
        """
        Select a prediction model.

        Returns:
            next_state, reward, done, info
        """

        if self.current_index is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        if action not in ACTIONS:
            raise ValueError(
                f"Invalid action: {action}"
            )

        row = self.data.iloc[
            self.current_index
        ]

        predicted_result = str(
            row[PREDICTION_COLUMNS[action]]
        )

        actual_result = str(
            row["FTR"]
        )

        reward = float(
            predicted_result == actual_result
        )

        info = {
            "SelectedModel": action,
            "Prediction": predicted_result,
            "ActualResult": actual_result,
            "Correct": bool(reward),
        }

        # Fixture decision ends immediately
        done = True

        self.current_index = None

        return None, reward, done, info

    def __len__(self):
        return len(self.data)

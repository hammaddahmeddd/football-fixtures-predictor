
"""
Football Fixtures Result Predictor
Q-Learning Reinforcement Learning Module

Purpose:
    Learn which classification model to select
    for different football match conditions.

Actions:
    0 - KNN
    1 - SVM
    2 - Decision Tree
    3 - Random Forest

Reward:
    +1 for a correct prediction
     0 for an incorrect prediction

Training:
    Base classifiers: 2021-22 to 2023-24
    Q-Learning:       2024-25
    Final testing:    2025-26

This is a one-step Q-Learning environment.
Gamma = 0 because each match ends the episode.
"""

from collections import defaultdict

import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
)

from src.feature_engineering import FEATURE_COLUMNS

from src.reinforcement_learning.environment import (
    ACTIONS,
    PREDICTION_COLUMNS,
    CONFIDENCE_COLUMNS,
    FootballStrategyEnvironment,
    create_state,
)


# =====================================================
# CONFIGURATION
# =====================================================

RANDOM_SEED = 42

VALIDATION_SEASON = "2024-25"
TEST_SEASON = "2025-26"

EPISODES = 150

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.0

EPSILON_START = 0.35
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.97

RESULT_LABELS = ["H", "D", "A"]

# Fixed fallback when an unseen state is encountered
DEFAULT_ACTION = "KNN"


# =====================================================
# LOAD DATA
# =====================================================

def load_data():
    """
    Load feature-engineered historical football data.
    """

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

    required_columns = (
        FEATURE_COLUMNS
        + [
            "Date",
            "Season",
            "HomeTeam",
            "AwayTeam",
            "FTR",
        ]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values detected in dataset."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="raise"
    )

    df = df.sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    if not df["FTR"].isin(RESULT_LABELS).all():
        raise ValueError(
            "Invalid match outcome labels found."
        )

    print(
        f"Loaded {len(df)} historical matches."
    )

    return df


# =====================================================
# LOAD CLASSIFICATION MODELS
# =====================================================

def load_classifiers():
    """
    Load previously trained classification models.

    The models must have been trained only on
    the 2021-22 to 2023-24 seasons.
    """

    model_files = {
        "KNN": "knn.joblib",
        "SVM": "svm.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
    }

    classifiers = {}

    for model_name, filename in model_files.items():

        model_path = (
            MODELS_DIR / filename
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model: {filename}\n"
                "Run python -m src.train_models first."
            )

        classifiers[model_name] = joblib.load(
            model_path
        )

        print(
            f"Loaded classifier: {model_name}"
        )

    return classifiers


# =====================================================
# GENERATE MODEL PREDICTIONS
# =====================================================

def generate_predictions(df, classifiers):
    """
    Generate predictions and prediction confidence
    for each classification model.

    Only historical pre-match features are passed
    into the classifiers.
    """

    output = df[
        [
            "Date",
            "Season",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "FormDifference",
        ]
    ].copy()

    X = df[FEATURE_COLUMNS]

    for model_name, model in classifiers.items():

        prediction_column = (
            PREDICTION_COLUMNS[model_name]
        )

        confidence_column = (
            CONFIDENCE_COLUMNS[model_name]
        )

        predictions = model.predict(X)

        output[prediction_column] = predictions

        if not hasattr(
            model,
            "predict_proba"
        ):
            raise ValueError(
                f"{model_name} does not support "
                "probability predictions."
            )

        probabilities = model.predict_proba(X)

        model_classes = list(
            model.classes_
        )

        prediction_indices = np.array([
            model_classes.index(label)
            for label in predictions
        ])

        confidence_scores = probabilities[
            np.arange(len(predictions)),
            prediction_indices
        ]

        output[confidence_column] = (
            confidence_scores
        )

        print(
            f"Generated predictions: {model_name}"
        )

    return output.reset_index(
        drop=True
    )


# =====================================================
# Q-TABLE HELPERS
# =====================================================

def get_q_values(q_table, state):
    """
    Return Q-values for a state.

    If the state was never seen during training,
    use zeros instead of raising KeyError.
    """

    return q_table.get(
        state,
        np.zeros(
            len(ACTIONS),
            dtype=float
        )
    )


def select_action(
    q_table,
    state,
    rng,
    epsilon=0.0,
):
    """
    Epsilon-greedy selection.

    During training:
        epsilon > 0 permits exploration.

    During testing:
        epsilon = 0 uses the learned policy.
    """

    if rng.random() < epsilon:

        action_index = int(
            rng.integers(
                len(ACTIONS)
            )
        )

        return ACTIONS[action_index]

    q_values = get_q_values(
        q_table,
        state
    )

    best_action_index = int(
        np.argmax(q_values)
    )

    return ACTIONS[best_action_index]


# =====================================================
# TRAIN Q-LEARNING
# =====================================================

def train_q_learning(validation_df):
    """
    Train Q-Learning using validation-season matches.

    Each match is a one-step episode.

    The training season is reused for several passes
    to estimate state-action reward values.

    The held-out test season is never used here.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    q_table = defaultdict(
        lambda: np.zeros(
            len(ACTIONS),
            dtype=float
        )
    )

    environment = FootballStrategyEnvironment(
        validation_df
    )

    epsilon = EPSILON_START

    training_history = []

    cumulative_reward = 0.0

    print(
        "\n========== Q-LEARNING TRAINING =========="
    )

    print(
        "Training matches:",
        len(environment)
    )

    print(
        "Available actions:",
        ACTIONS
    )

    print(
        "Training passes:",
        EPISODES
    )

    for episode in range(EPISODES):

        episode_reward = 0.0

        match_order = rng.permutation(
            len(environment)
        )

        for match_index in match_order:

            state = environment.reset(
                int(match_index)
            )

            selected_action = select_action(
                q_table=q_table,
                state=state,
                rng=rng,
                epsilon=epsilon,
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = environment.step(
                selected_action
            )

            action_index = ACTIONS.index(
                selected_action
            )

            old_q_value = q_table[state][
                action_index
            ]

            # One-step Q-Learning.
            # The future reward is zero because
            # the episode terminates after one match.
            target = (
                reward
                + DISCOUNT_FACTOR * 0.0
            )

            updated_q_value = (
                old_q_value
                + LEARNING_RATE
                * (
                    target
                    - old_q_value
                )
            )

            q_table[state][
                action_index
            ] = updated_q_value

            episode_reward += reward

        cumulative_reward += episode_reward

        # Evaluate on training data for
        # convergence visualization only.
        training_metrics = evaluate_policy(
            dataframe=validation_df,
            q_table=q_table,
        )

        training_history.append({
            "Episode": episode + 1,
            "TrainingReward": episode_reward,
            "CumulativeTrainingReward":
                cumulative_reward,
            "GreedyValidationAccuracy":
                training_metrics["Accuracy"],
            "Epsilon": epsilon,
        })

        epsilon = max(
            EPSILON_MIN,
            epsilon * EPSILON_DECAY
        )

        if (
            episode == 0
            or (episode + 1) % 25 == 0
        ):

            print(
                f"Episode {episode + 1:3d} | "
                f"Reward: {episode_reward:.0f} | "
                f"Accuracy: "
                f"{training_metrics['Accuracy']:.4f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    print("\nQ-Learning training completed.")

    print(
        "Total learned states:",
        len(q_table)
    )

    return (
        dict(q_table),
        pd.DataFrame(
            training_history
        )
    )


# =====================================================
# EVALUATE Q-LEARNING
# =====================================================

def evaluate_policy(
    dataframe,
    q_table,
    return_decisions=False,
):
    """
    Evaluate the frozen Q-Learning policy.

    No Q-table updates occur during evaluation.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    environment = FootballStrategyEnvironment(
        dataframe
    )

    decision_records = []

    for match_index in range(
        len(environment)
    ):

        state = environment.reset(
            match_index
        )

        selected_action = select_action(
            q_table=q_table,
            state=state,
            rng=rng,
            epsilon=0.0,
        )

        (
            next_state,
            reward,
            done,
            info,
        ) = environment.step(
            selected_action
        )

        source_row = dataframe.iloc[
            match_index
        ]

        selected_confidence = float(
            source_row[
                CONFIDENCE_COLUMNS[selected_action]
            ]
        )

        current_q_values = get_q_values(
            q_table,
            state
        )

        learned_q_value = float(
            current_q_values[
                ACTIONS.index(
                    selected_action
                )
            ]
        )

        decision_records.append({
            "Date": source_row["Date"],
            "HomeTeam": source_row["HomeTeam"],
            "AwayTeam": source_row["AwayTeam"],
            "State": str(state),
            "SelectedModel": selected_action,
            "Prediction": info["Prediction"],
            "ActualResult": info["ActualResult"],
            "SelectedConfidence":
                selected_confidence,
            "LearnedQValue":
                learned_q_value,
            "Reward": reward,
            "Correct": int(reward),
            "ModelsDisagree": bool(state[0]),
        })

        if not done:
            raise RuntimeError(
                "Expected a one-step terminal episode."
            )

    decisions_df = pd.DataFrame(
        decision_records
    )

    accuracy = accuracy_score(
        decisions_df["ActualResult"],
        decisions_df["Prediction"]
    )

    macro_f1 = f1_score(
        decisions_df["ActualResult"],
        decisions_df["Prediction"],
        labels=RESULT_LABELS,
        average="macro",
        zero_division=0
    )

    results = {
        "Accuracy": float(accuracy),
        "MacroF1": float(macro_f1),
        "CumulativeReward": float(
            decisions_df["Reward"].sum()
        ),
    }

    if return_decisions:
        return results, decisions_df

    return results


# =====================================================
# BASELINE COMPARISON
# =====================================================

def evaluate_fixed_models(dataframe):
    """
    Evaluate all individual classifiers and
    an always-home prediction baseline.
    """

    comparison_results = []

    actual_results = dataframe["FTR"]

    for model_name in ACTIONS:

        predictions = dataframe[
            PREDICTION_COLUMNS[model_name]
        ]

        comparison_results.append({
            "Strategy": model_name,
            "Accuracy": accuracy_score(
                actual_results,
                predictions
            ),
            "MacroF1": f1_score(
                actual_results,
                predictions,
                labels=RESULT_LABELS,
                average="macro",
                zero_division=0
            ),
        })

    always_home_predictions = np.full(
        len(dataframe),
        "H"
    )

    comparison_results.append({
        "Strategy": "Always Home",
        "Accuracy": accuracy_score(
            actual_results,
            always_home_predictions
        ),
        "MacroF1": f1_score(
            actual_results,
            always_home_predictions,
            labels=RESULT_LABELS,
            average="macro",
            zero_division=0
        ),
    })

    return pd.DataFrame(
        comparison_results
    )


# =====================================================
# MAJORITY VOTING
# =====================================================

def majority_vote_prediction(
    row,
    fallback_model,
):
    """
    Combine four classifier predictions using
    majority voting.

    A validation-selected classifier resolves ties.
    """

    predictions = [
        row[
            PREDICTION_COLUMNS[model_name]
        ]
        for model_name in ACTIONS
    ]

    prediction_counts = {
        label: predictions.count(label)
        for label in RESULT_LABELS
    }

    highest_count = max(
        prediction_counts.values()
    )

    tied_labels = [
        label
        for label, count
        in prediction_counts.items()
        if count == highest_count
    ]

    if len(tied_labels) == 1:
        return tied_labels[0]

    return row[
        PREDICTION_COLUMNS[fallback_model]
    ]


# =====================================================
# SAVE GRAPHS
# =====================================================

def save_graphs(
    training_history,
    test_decisions,
):
    """
    Generate reward and convergence plots.
    """

    graphs_directory = (
        OUTPUTS_DIR / "graphs"
    )

    graphs_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Graph 1: Cumulative training reward

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        training_history["Episode"],
        training_history[
            "CumulativeTrainingReward"
        ],
        linewidth=2
    )

    ax.set_title(
        "Q-Learning Cumulative Training Reward"
    )

    ax.set_xlabel(
        "Training Pass"
    )

    ax.set_ylabel(
        "Cumulative Reward"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        graphs_directory
        / "q_learning_cumulative_reward.png",
        dpi=150
    )

    plt.close(fig)

    # Graph 2: Training policy convergence

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        training_history["Episode"],
        training_history[
            "GreedyValidationAccuracy"
        ],
        linewidth=2
    )

    ax.set_title(
        "Q-Learning Training Policy Convergence"
    )

    ax.set_xlabel(
        "Training Pass"
    )

    ax.set_ylabel(
        "Greedy Accuracy on RL Training Data"
    )

    ax.set_ylim(
        0, 1
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        graphs_directory
        / "q_learning_convergence.png",
        dpi=150
    )

    plt.close(fig)

    # Graph 3: Held-out test cumulative reward

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    cumulative_test_rewards = (
        test_decisions["Reward"].cumsum()
    )

    ax.plot(
        np.arange(
            1,
            len(test_decisions) + 1
        ),
        cumulative_test_rewards,
        linewidth=2,
        label="Q-Learning"
    )

    ax.set_title(
        "Q-Learning Held-Out Test Cumulative Reward"
    )

    ax.set_xlabel(
        "Test Matches"
    )

    ax.set_ylabel(
        "Correct Predictions"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        graphs_directory
        / "q_learning_test_rewards.png",
        dpi=150
    )

    plt.close(fig)

    print("\nQ-Learning graphs saved.")


# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

def main():

    print(
        "========== FOOTBALL Q-LEARNING =========="
    )

    # Step 1: Load historical football data

    football_data = load_data()

    # Step 2: Load trained classification models

    classifiers = load_classifiers()

    # Step 3: Split RL training and test seasons

    validation_data = football_data[
        football_data["Season"]
        == VALIDATION_SEASON
    ].copy()

    test_data = football_data[
        football_data["Season"]
        == TEST_SEASON
    ].copy()

    if (
        len(validation_data) == 0
        or len(test_data) == 0
    ):
        raise ValueError(
            "Validation or test season is empty."
        )

    # Step 4: Generate predictions from base models

    validation_predictions = generate_predictions(
        validation_data,
        classifiers
    )

    test_predictions = generate_predictions(
        test_data,
        classifiers
    )

    print(
        "\nRL training matches:",
        len(validation_predictions)
    )

    print(
        "RL testing matches:",
        len(test_predictions)
    )

    # Step 5: Train Q-Learning policy

    (
        q_table,
        training_history,
    ) = train_q_learning(
        validation_predictions
    )

    # Step 6: Evaluate frozen policy on test season

    (
        test_metrics,
        test_decisions,
    ) = evaluate_policy(
        dataframe=test_predictions,
        q_table=q_table,
        return_decisions=True
    )

    print(
        "\n========== HELD-OUT TEST RESULTS =========="
    )

    print(
        "Q-Learning Accuracy:",
        f"{test_metrics['Accuracy']:.4f}"
    )

    print(
        "Q-Learning Macro F1:",
        f"{test_metrics['MacroF1']:.4f}"
    )

    print(
        "Cumulative Test Reward:",
        test_metrics["CumulativeReward"]
    )

    # Step 7: Compare with fixed classifiers

    comparison = evaluate_fixed_models(
        test_predictions
    )

    q_learning_row = pd.DataFrame([
        {
            "Strategy": "Q-Learning",
            "Accuracy": test_metrics["Accuracy"],
            "MacroF1": test_metrics["MacroF1"],
        }
    ])

    comparison = pd.concat(
        [
            comparison,
            q_learning_row,
        ],
        ignore_index=True
    )

    # Step 8: Select majority-vote tie-breaker
    # using validation-season performance only

    validation_scores = evaluate_fixed_models(
        validation_predictions
    )

    validation_experts = validation_scores[
        validation_scores["Strategy"].isin(
            ACTIONS
        )
    ]

    fallback_model = (
        validation_experts
        .sort_values(
            ["MacroF1", "Accuracy"],
            ascending=False
        )
        .iloc[0]["Strategy"]
    )

    # Step 9: Evaluate majority voting

    majority_predictions = test_predictions.apply(
        lambda row: majority_vote_prediction(
            row,
            fallback_model
        ),
        axis=1
    )

    majority_accuracy = accuracy_score(
        test_predictions["FTR"],
        majority_predictions
    )

    majority_f1 = f1_score(
        test_predictions["FTR"],
        majority_predictions,
        labels=RESULT_LABELS,
        average="macro",
        zero_division=0
    )

    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame([
                {
                    "Strategy": "Majority Vote",
                    "Accuracy": majority_accuracy,
                    "MacroF1": majority_f1,
                }
            ])
        ],
        ignore_index=True
    )

    print(
        "\n========== STRATEGY COMPARISON =========="
    )

    print(
        comparison
        .sort_values(
            "MacroF1",
            ascending=False
        )
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # Step 10: Save learned policy

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    policy_data = {
        "q_table": q_table,
        "actions": ACTIONS,
        "gamma": DISCOUNT_FACTOR,
        "learning_rate": LEARNING_RATE,
        "training_season": VALIDATION_SEASON,
        "test_season": TEST_SEASON,
        "fallback_model": fallback_model,
        "default_action": DEFAULT_ACTION,
    }

    joblib.dump(
        policy_data,
        MODELS_DIR / "q_learning_policy.joblib"
    )

    # Step 11: Save evaluation results

    metrics_directory = (
        OUTPUTS_DIR / "metrics"
    )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        metrics_directory
        / "q_learning_comparison.csv",
        index=False
    )

    training_history.to_csv(
        metrics_directory
        / "q_learning_training_history.csv",
        index=False
    )

    test_decisions.to_csv(
        metrics_directory
        / "q_learning_test_decisions.csv",
        index=False
    )

    validation_predictions.to_csv(
        metrics_directory
        / "rl_validation_predictions.csv",
        index=False
    )

    test_predictions.to_csv(
        metrics_directory
        / "rl_test_predictions.csv",
        index=False
    )

    # Step 12: Generate visualizations

    save_graphs(
        training_history,
        test_decisions
    )

    print(
        "\n========== Q-LEARNING COMPLETED =========="
    )

    print("Q-Learning policy saved.")
    print("Model comparison saved.")
    print("Prediction decisions saved.")
    print("Training history saved.")
    print("Convergence and reward graphs saved.")


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    main()

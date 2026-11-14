
import pandas as pd
import streamlit as st

from config import (
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
)


METRICS_DIR = OUTPUTS_DIR / "metrics"
GRAPH_DIR = OUTPUTS_DIR / "graphs"


@st.cache_data
def read_optional_csv(path):

    if not path.exists():
        return None

    return pd.read_csv(path)


def render():

    st.title("🧠 Q-Learning Strategy Maker")

    st.write(
        "A reinforcement learning component that "
        "selects one of four trained classification "
        "models for each match."
    )

    st.info(
        "The agent chooses a prediction model, "
        "not an on-field football tactic."
    )

    st.subheader("Available Actions")

    st.dataframe(
        pd.DataFrame({
            "Action": [
                "KNN",
                "SVM",
                "Decision Tree",
                "Random Forest",
            ],
            "Meaning": [
                "Use KNN prediction",
                "Use SVM prediction",
                "Use Decision Tree prediction",
                "Use Random Forest prediction",
            ],
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Q-Learning Evaluation")

    comparison = read_optional_csv(
        METRICS_DIR
        / "q_learning_comparison.csv"
    )

    if comparison is not None:

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.warning(
            "Q-Learning comparison data not found."
        )

    st.divider()

    st.subheader("Training History")

    history = read_optional_csv(
        METRICS_DIR
        / "q_learning_training_history.csv"
    )

    if history is not None:

        st.dataframe(
            history.tail(10),
            hide_index=True,
            use_container_width=True,
        )

        numeric_columns = [
            col for col in history.columns
            if pd.api.types.is_numeric_dtype(
                history[col]
            )
        ]

        # Prefer a reward or accuracy column.
        metric_columns = [
            col for col in numeric_columns
            if (
                "reward" in col.lower()
                or "accuracy" in col.lower()
            )
        ]

        if metric_columns:

            st.line_chart(
                history[metric_columns],
                y_label="Training Metric"
            )

    else:

        st.info(
            "Training history is not available."
        )

    st.divider()

    st.subheader("Weekly Strategy Decisions")

    weekly = read_optional_csv(
        PREDICTIONS_DIR
        / "weekly_predictions.csv"
    )

    if weekly is not None:

        columns = [
            "HomeTeam",
            "AwayTeam",
            "SelectedModel",
            "PredictedResult",
            "SelectedConfidence",
        ]

        available = [
            col for col in columns
            if col in weekly.columns
        ]

        st.dataframe(
            weekly[available],
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Models Selected This Week")

        selections = (
            weekly["SelectedModel"]
            .value_counts()
            .rename_axis("Model")
            .reset_index(name="Matches")
        )

        st.bar_chart(
            selections.set_index("Model"),
            y_label="Number of Matches",
        )

    st.divider()

    st.subheader("How the Agent Works")

    st.write(
        "1. The agent observes disagreement "
        "between models, prediction confidence "
        "and the difference in recent team form."
    )

    st.write(
        "2. It chooses KNN, SVM, Decision Tree "
        "or Random Forest."
    )

    st.write(
        "3. During training, it receives a reward "
        "of 1 for a correct prediction and "
        "0 for an incorrect prediction."
    )

    st.write(
        "4. It learns action values for the "
        "observed states and uses the highest "
        "value during prediction."
    )

    st.caption(
        "This is a one-step contextual decision "
        "problem implemented with Q-Learning. "
        "The saved policy is evaluated on a "
        "later held-out season."
    )

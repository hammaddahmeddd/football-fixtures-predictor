# V4 CLEAN DASHBOARD ADAPTER
from dashboard import v4_data as v4

import pandas as pd
import streamlit as st

from config import OUTPUTS_DIR


METRICS_DIR = OUTPUTS_DIR / "metrics"
GRAPH_DIR = OUTPUTS_DIR / "graphs"


def load_results():
    return v4.metrics("Outcome")


def render():

    st.title("📊 Model Comparison")

    st.write(
        "Evaluation of classification "
        "and additional models on the 760-match historical benchmark."
    )

    df = load_results()

    if df is None or df.empty:

        st.warning(
            "Classification results are missing. "
            "Run python -m src.accuracy_v39."
        )
        return

    st.subheader("Saved Classification Results")

    st.dataframe(
        df,
        hide_index=True,
        width="stretch"
    )

    st.divider()

    # Different training scripts may use
    # different metric-column names.
    metric_candidates = {
        "Accuracy": [
            "Test_Accuracy",
            "Test Accuracy",
            "Accuracy",
            "accuracy",
        ],
        "Macro F1": [
            "Test_Macro_F1",
            "Test_MacroF1",
            "Macro_F1",
            "MacroF1",
            "Macro F1",
            "F1",
        ],
    }

    def find_column(candidates):

        for name in candidates:
            if name in df.columns:
                return name

        return None

    model_col = find_column([
        "Model",
        "model",
        "Model_Name",
        "Algorithm",
    ])

    accuracy_col = find_column(
        metric_candidates["Accuracy"]
    )

    f1_col = find_column(
        metric_candidates["Macro F1"]
    )

    if model_col and accuracy_col:

        st.subheader("Classification Accuracy")

        chart = df[
            [model_col, accuracy_col]
        ].copy()

        chart[accuracy_col] = pd.to_numeric(
            chart[accuracy_col],
            errors="coerce"
        )

        chart = chart.dropna()

        if not chart.empty:

            st.bar_chart(
                chart.set_index(model_col),
                y_label="Accuracy",
            )

    if model_col and f1_col:

        st.subheader("Macro F1 Score")

        chart = df[
            [model_col, f1_col]
        ].copy()

        chart[f1_col] = pd.to_numeric(
            chart[f1_col],
            errors="coerce"
        )

        chart = chart.dropna()

        if not chart.empty:

            st.bar_chart(
                chart.set_index(model_col),
                y_label="Macro F1",
            )

    st.divider()

    st.subheader("Confusion Matrices (legacy V2)")

    images = sorted(
        GRAPH_DIR.glob("*confusion*.png")
    )

    if images:

        for image_path in images:

            st.image(
                str(image_path),
                caption=image_path.stem.replace(
                    "_", " "
                ).title(),
                width="stretch",
            )

    else:

        st.info(
            "No confusion-matrix images found."
        )

    st.divider()

    st.subheader("Evaluation Notes")

    st.write(
        "Models were developed using a chronological "
        "split so later-season test results were "
        "not included in the earlier training data."
    )

    st.write(
        "Accuracy measures the proportion of "
        "correct outcomes. Macro F1 gives equal "
        "weight to home wins, draws and away wins."
    )

    st.caption(
        "A model that mostly predicts the most "
        "common result can achieve reasonable "
        "accuracy while performing poorly on "
        "minority classes."
    )


    v4.research()

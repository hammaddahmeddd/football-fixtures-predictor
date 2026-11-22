# V4 CLEAN DASHBOARD ADAPTER
from dashboard import v4_data as v4

"""
Dashboard: Prediction History and Live Accuracy
"""

import pandas as pd
import streamlit as st

from config import PREDICTIONS_DIR, OUTPUTS_DIR


HISTORY_PATH = (
    PREDICTIONS_DIR / "prediction_history.csv"
)

SUMMARY_PATH = (
    OUTPUTS_DIR / "metrics" /
    "live_accuracy_summary.csv"
)

RESULT_NAMES = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}


def format_accuracy(value):

    if pd.isna(value):
        return "N/A"

    return f"{float(value):.1%}"


def render():
    v4.history_page()



    

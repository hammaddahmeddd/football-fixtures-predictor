from collections import defaultdict
import numpy as np
import pandas as pd
from src.feature_engineering import (
    feature_engineering, FEATURE_COLUMNS, get_advanced_team_stats,
    update_elo, add_advanced_fixture_features,
)


def test_no_same_day_results_in_features():
    df = pd.DataFrame([
        {"Date": "2025-08-01", "Season": "2025-26", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 9, "FTAG": 0, "FTR": "H"},
        {"Date": "2025-08-01", "Season": "2025-26", "HomeTeam": "C", "AwayTeam": "D", "FTHG": 8, "FTAG": 0, "FTR": "H"},
        {"Date": "2025-08-08", "Season": "2025-26", "HomeTeam": "A", "AwayTeam": "C", "FTHG": 1, "FTAG": 1, "FTR": "D"},
    ])
    result = feature_engineering(df)
    first = result[result.Date == pd.Timestamp('2025-08-01')]
    assert (first.HomeMatchesSeen == 0).all()
    assert (first.AwayMatchesSeen == 0).all()
    assert (first.EloDifference == 0).all()
    later = result[result.Date == pd.Timestamp('2025-08-08')].iloc[0]
    assert later.HomeMatchesSeen == 1
    assert later.AwayMatchesSeen == 1
    assert later.HomeElo != 1500
    assert not result[FEATURE_COLUMNS].isna().any().any()


def test_elo_is_zero_sum_per_match():
    h, a = update_elo(1500, 1500, 2, 1)
    assert abs(h + a - 3000) < 1e-8
    assert h > a

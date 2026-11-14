
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR


# Features that will be used by ML models
FEATURE_COLUMNS = [
    "HomeLast5Points",
    "AwayLast5Points",
    "HomeAvgGoalsScored",
    "AwayAvgGoalsScored",
    "HomeAvgGoalsConceded",
    "AwayAvgGoalsConceded",
    "HomeWinRate",
    "AwayWinRate",
    "HomeGoalDifference",
    "AwayGoalDifference",
    "HomeHomeWinRate",
    "AwayAwayWinRate",
    "HomeRestDays",
    "AwayRestDays",
    "RestDaysDifference",
    "FormDifference",
    "GoalDifferenceDifference",
]


def load_cleaned_data():
    """Load the cleaned Premier League dataset."""

    file_path = (
        PROCESSED_DATA_DIR
        / "premier_league_cleaned.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            "Run python -m src.preprocessing first."
        )

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    print(f"Loaded {len(df)} matches")

    return df


def get_team_statistics(
    history,
    current_date,
    venue,
    last_n=5,
):
    """
    Calculate pre-match statistics for one team.

    history contains only previously completed games.
    """

    recent = history[-last_n:]

    if not recent:
        return {
            "Last5Points": 0.0,
            "AvgGoalsScored": 0.0,
            "AvgGoalsConceded": 0.0,
            "WinRate": 0.0,
            "GoalDifference": 0.0,
            "VenueWinRate": 0.0,
            "RestDays": 7.0,
        }

    matches_played = len(recent)

    points = sum(
        match["points"] for match in recent
    )

    goals_scored = sum(
        match["goals_scored"] for match in recent
    )

    goals_conceded = sum(
        match["goals_conceded"] for match in recent
    )

    wins = sum(
        match["points"] == 3 for match in recent
    )

    # Previous games at the relevant venue
    venue_history = [
        match for match in history
        if match["venue"] == venue
    ][-last_n:]

    venue_win_rate = (
        sum(m["points"] == 3 for m in venue_history)
        / len(venue_history)
        if venue_history else 0.0
    )

    last_match_date = history[-1]["date"]

    rest_days = (
        current_date - last_match_date
    ).days

    return {
        "Last5Points": float(points),
        "AvgGoalsScored": goals_scored / matches_played,
        "AvgGoalsConceded": goals_conceded / matches_played,
        "WinRate": wins / matches_played,
        "GoalDifference": (
            goals_scored - goals_conceded
        ) / matches_played,
        "VenueWinRate": venue_win_rate,
        "RestDays": float(
            max(0, min(rest_days, 30))
        ),
    }


def feature_engineering(df):
    """
    Generate features using only historical matches.

    Matches on the same date are processed together.
    Their results are added to history only after
    every match on that date has been assigned features.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    team_history = defaultdict(list)

    features = []

    print("\nGenerating pre-match features...")

    for match_date, day_matches in df.groupby(
        "Date", sort=True
    ):

        # First calculate features for all games
        # on this date, without seeing today's results.
        for index, match in day_matches.iterrows():

            home_team = match["HomeTeam"]
            away_team = match["AwayTeam"]

            home_stats = get_team_statistics(
                team_history[home_team],
                match_date,
                venue="Home",
            )

            away_stats = get_team_statistics(
                team_history[away_team],
                match_date,
                venue="Away",
            )

            row_features = {
                "HomeLast5Points":
                    home_stats["Last5Points"],

                "AwayLast5Points":
                    away_stats["Last5Points"],

                "HomeAvgGoalsScored":
                    home_stats["AvgGoalsScored"],

                "AwayAvgGoalsScored":
                    away_stats["AvgGoalsScored"],

                "HomeAvgGoalsConceded":
                    home_stats["AvgGoalsConceded"],

                "AwayAvgGoalsConceded":
                    away_stats["AvgGoalsConceded"],

                "HomeWinRate":
                    home_stats["WinRate"],

                "AwayWinRate":
                    away_stats["WinRate"],

                "HomeGoalDifference":
                    home_stats["GoalDifference"],

                "AwayGoalDifference":
                    away_stats["GoalDifference"],

                "HomeHomeWinRate":
                    home_stats["VenueWinRate"],

                "AwayAwayWinRate":
                    away_stats["VenueWinRate"],

                "HomeRestDays":
                    home_stats["RestDays"],

                "AwayRestDays":
                    away_stats["RestDays"],

                "RestDaysDifference":
                    home_stats["RestDays"]
                    - away_stats["RestDays"],

                "FormDifference":
                    home_stats["Last5Points"]
                    - away_stats["Last5Points"],

                "GoalDifferenceDifference":
                    home_stats["GoalDifference"]
                    - away_stats["GoalDifference"],
            }

            features.append((index, row_features))

        # Only now update the histories with
        # the actual results from this date.
        for _, match in day_matches.iterrows():

            home_team = match["HomeTeam"]
            away_team = match["AwayTeam"]

            home_goals = int(match["FTHG"])
            away_goals = int(match["FTAG"])

            if home_goals > away_goals:
                home_points, away_points = 3, 0
            elif home_goals < away_goals:
                home_points, away_points = 0, 3
            else:
                home_points, away_points = 1, 1

            team_history[home_team].append({
                "date": match_date,
                "venue": "Home",
                "goals_scored": home_goals,
                "goals_conceded": away_goals,
                "points": home_points,
            })

            team_history[away_team].append({
                "date": match_date,
                "venue": "Away",
                "goals_scored": away_goals,
                "goals_conceded": home_goals,
                "points": away_points,
            })

    # Join feature rows back to their original matches
    feature_df = pd.DataFrame(
        {
            index: values
            for index, values in features
        }
    ).T

    feature_df = feature_df.reindex(df.index)

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            feature_df[column],
            errors="coerce"
        )

    # Check features contain valid numbers
    if df[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError(
            "Feature engineering produced missing values."
        )

    print("Feature engineering completed!")
    print(f"Features created: {len(FEATURE_COLUMNS)}")

    return df


def save_feature_data(df):
    """Save the final feature dataset."""

    output_path = (
        PROCESSED_DATA_DIR
        / "premier_league_features.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to: {output_path}")

    return output_path


def main():

    df = load_cleaned_data()

    featured_df = feature_engineering(df)

    save_feature_data(featured_df)

    print("\n========== FEATURE SUMMARY ==========")

    print("Total matches:", len(featured_df))

    print("Total columns:", len(featured_df.columns))

    print("New features:", len(FEATURE_COLUMNS))

    print("\nFirst five matches with features:")

    print(
        featured_df[
            [
                "HomeTeam",
                "AwayTeam",
                "HomeLast5Points",
                "AwayLast5Points",
                "FormDifference",
            ]
        ].head()
    )

    print("\nFeature missing values:")

    print(
        featured_df[FEATURE_COLUMNS]
        .isnull()
        .sum()
    )


if __name__ == "__main__":
    main()

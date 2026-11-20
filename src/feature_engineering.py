
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR


# Features that will be used by ML models
LEGACY_FEATURE_COLUMNS = [
    "HomeLast5Points", "AwayLast5Points",
    "HomeAvgGoalsScored", "AwayAvgGoalsScored",
    "HomeAvgGoalsConceded", "AwayAvgGoalsConceded",
    "HomeWinRate", "AwayWinRate",
    "HomeGoalDifference", "AwayGoalDifference",
    "HomeHomeWinRate", "AwayAwayWinRate",
    "HomeRestDays", "AwayRestDays", "RestDaysDifference",
    "FormDifference", "GoalDifferenceDifference",
]

ADVANCED_FEATURE_COLUMNS = [
    "HomeLast3Points", "AwayLast3Points",
    "HomeLast10PointsPerGame", "AwayLast10PointsPerGame",
    "HomeLast10GoalsFor", "AwayLast10GoalsFor",
    "HomeLast10GoalsAgainst", "AwayLast10GoalsAgainst",
    "HomeWeightedPoints", "AwayWeightedPoints",
    "HomeHomeGoalsFor", "AwayAwayGoalsFor",
    "HomeHomeGoalsAgainst", "AwayAwayGoalsAgainst",
    "HomeMatchesSeen", "AwayMatchesSeen",
    "HomeElo", "AwayElo", "EloDifference",
    "RecentGoalsScoredDifference", "RecentGoalsConcededDifference",
]

FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS + ADVANCED_FEATURE_COLUMNS

ELO_INITIAL = 1500.0
ELO_HOME_BONUS = 65.0
ELO_K = 20.0


def elo_expectation(home_rating, away_rating):
    return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - ELO_HOME_BONUS) / 400.0))


def update_elo(home_rating, away_rating, home_goals, away_goals):
    expected = elo_expectation(home_rating, away_rating)
    actual = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
    goal_factor = 1.0 + 0.25 * min(abs(home_goals - away_goals), 4)
    delta = ELO_K * goal_factor * (actual - expected)
    return home_rating + delta, away_rating - delta


def get_advanced_team_stats(history, venue):
    """All values are based exclusively on already-completed matches."""
    def stats(matches):
        count = len(matches)
        return (
            sum(m["points"] for m in matches) / count if count else 1.0,
            sum(m["goals_scored"] for m in matches) / count if count else 1.3,
            sum(m["goals_conceded"] for m in matches) / count if count else 1.3,
        )

    recent3 = history[-3:]
    recent10 = history[-10:]
    venue_matches = [m for m in history if m["venue"] == venue][-5:]
    pts10, gf10, ga10 = stats(recent10)
    _, venue_gf, venue_ga = stats(venue_matches)
    recent = history[-5:]
    weights = np.array([0.78 ** (len(recent) - 1 - i) for i in range(len(recent))], dtype=float)
    weighted_points = float(np.average([m["points"] for m in recent], weights=weights)) if recent else 1.0
    return {
        "Last3Points": float(sum(m["points"] for m in recent3)),
        "Last10PointsPerGame": float(pts10),
        "Last10GoalsFor": float(gf10),
        "Last10GoalsAgainst": float(ga10),
        "WeightedPoints": weighted_points,
        "VenueGoalsFor": float(venue_gf),
        "VenueGoalsAgainst": float(venue_ga),
        "MatchesSeen": float(min(len(history), 38)),
    }


def add_advanced_fixture_features(row, home_history, away_history, home_elo, away_elo):
    h = get_advanced_team_stats(home_history, "Home")
    a = get_advanced_team_stats(away_history, "Away")
    for label in ["Last3Points", "Last10PointsPerGame", "Last10GoalsFor", "Last10GoalsAgainst", "WeightedPoints", "MatchesSeen"]:
        row[f"Home{label}"] = h[label]
        row[f"Away{label}"] = a[label]
    row["HomeHomeGoalsFor"] = h["VenueGoalsFor"]
    row["AwayAwayGoalsFor"] = a["VenueGoalsFor"]
    row["HomeHomeGoalsAgainst"] = h["VenueGoalsAgainst"]
    row["AwayAwayGoalsAgainst"] = a["VenueGoalsAgainst"]
    row["HomeElo"] = float(home_elo)
    row["AwayElo"] = float(away_elo)
    row["EloDifference"] = float(home_elo - away_elo)
    row["RecentGoalsScoredDifference"] = h["Last10GoalsFor"] - a["Last10GoalsFor"]
    row["RecentGoalsConcededDifference"] = h["Last10GoalsAgainst"] - a["Last10GoalsAgainst"]
    return row


def elo_ratings_before(results, before_date):
    """Calculate ratings using matches with Date strictly before before_date."""
    ratings = defaultdict(lambda: ELO_INITIAL)
    eligible = results[pd.to_datetime(results["Date"]) < pd.Timestamp(before_date)]
    for date, group in eligible.sort_values("Date").groupby("Date", sort=True):
        # Freeze pre-match ratings for all fixtures on the same date.
        deltas = []
        for _, match in group.iterrows():
            h, a = match["HomeTeam"], match["AwayTeam"]
            new_h, new_a = update_elo(ratings[h], ratings[a], int(match["FTHG"]), int(match["FTAG"]))
            deltas.append((h, a, new_h - ratings[h], new_a - ratings[a]))
        for h, a, dh, da in deltas:
            ratings[h] += dh
            ratings[a] += da
    return ratings



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
    elo_ratings = defaultdict(lambda: ELO_INITIAL)

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

            add_advanced_fixture_features(
                row_features,
                team_history[home_team],
                team_history[away_team],
                elo_ratings[home_team],
                elo_ratings[away_team],
            )
            features.append((index, row_features))

        # Only now update the histories with
        # the actual results from this date.
        elo_updates = []
        for _, match in day_matches.iterrows():
            h, a = match["HomeTeam"], match["AwayTeam"]
            new_h, new_a = update_elo(
                elo_ratings[h], elo_ratings[a],
                int(match["FTHG"]), int(match["FTAG"]),
            )
            elo_updates.append((h, a, new_h - elo_ratings[h], new_a - elo_ratings[a]))
        for h, a, dh, da in elo_updates:
            elo_ratings[h] += dh
            elo_ratings[a] += da

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


"""
Football Fixtures Result Predictor
Step 9: Weekly Premier League Predictions

Run:
    python -m src.predictions

Features:
- Historical Premier League data
- Current-season results
- Upcoming fixture API
- Comprehensive team name normalization
- Pre-match feature engineering
- Four classification models
- Goal regression
- Q-Learning strategy selection
- CSV prediction export
"""

from datetime import datetime, timezone
from io import StringIO
from difflib import get_close_matches
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import requests

from config import (
    RAW_DATA_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    FOOTBALL_API_KEY,
    FOOTBALL_DATA_URL,
    CURRENT_SEASON,
    CURRENT_SEASON_START_YEAR,
    FOOTBALL_API_BASE_URL,
)

from src.feature_engineering import (
    FEATURE_COLUMNS,
    get_team_statistics,
    add_advanced_fixture_features,
    elo_ratings_before,
    LEGACY_FEATURE_COLUMNS,
)

from src.reinforcement_learning.environment import (
    ACTIONS,
    PREDICTION_COLUMNS,
    CONFIDENCE_COLUMNS,
    create_state,
)


# =====================================================
# CONFIGURATION
# =====================================================

LEAGUE_CODE = "E0"

RESULT_LABELS = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}

REQUEST_TIMEOUT = 30


# =====================================================
# TEAM NAME ALIASES
# =====================================================

# The values must match the historical CSV names.

TEAM_ALIASES = {

    # Arsenal
    "Arsenal": "Arsenal",
    "Arsenal FC": "Arsenal",

    # Aston Villa
    "Aston Villa": "Aston Villa",
    "Aston Villa FC": "Aston Villa",

    # Bournemouth
    "Bournemouth": "Bournemouth",
    "AFC Bournemouth": "Bournemouth",

    # Brentford
    "Brentford": "Brentford",
    "Brentford FC": "Brentford",

    # Brighton
    "Brighton": "Brighton",
    "Brighton Hove": "Brighton",
    "Brighton and Hove": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Brighton & Hove Albion FC": "Brighton",

    # Burnley
    "Burnley": "Burnley",
    "Burnley FC": "Burnley",

    # Chelsea
    "Chelsea": "Chelsea",
    "Chelsea FC": "Chelsea",

    # Coventry City
    "Coventry": "Coventry City",
    "Coventry City": "Coventry City",
    "Coventry City FC": "Coventry City",

    # Crystal Palace
    "Crystal Palace": "Crystal Palace",
    "Crystal Palace FC": "Crystal Palace",

    # Everton
    "Everton": "Everton",
    "Everton FC": "Everton",

    # Fulham
    "Fulham": "Fulham",
    "Fulham FC": "Fulham",

    # Hull City
    "Hull": "Hull City",
    "Hull City": "Hull City",
    "Hull City AFC": "Hull City",

    # Ipswich
    "Ipswich": "Ipswich",
    "Ipswich Town": "Ipswich",
    "Ipswich Town FC": "Ipswich",

    # Leeds United
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Leeds United FC": "Leeds",

    # Leicester City
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Leicester City FC": "Leicester",

    # Liverpool
    "Liverpool": "Liverpool",
    "Liverpool FC": "Liverpool",

    # Luton Town
    "Luton": "Luton",
    "Luton Town": "Luton",
    "Luton Town FC": "Luton",

    # Manchester City
    "Man City": "Man City",
    "Manchester City": "Man City",
    "Manchester City FC": "Man City",
    "Man. City": "Man City",

    # Manchester United
    "Man United": "Man United",
    "Man Utd": "Man United",
    "Manchester United": "Man United",
    "Manchester United FC": "Man United",
    "Manchester Utd": "Man United",
    "Man. United": "Man United",

    # Newcastle
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Newcastle United FC": "Newcastle",

    # Nottingham Forest
    "Nottingham": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Nottingham Forest FC": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",

    # Norwich City
    "Norwich": "Norwich",
    "Norwich City": "Norwich",
    "Norwich City FC": "Norwich",

    # Sheffield United
    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",
    "Sheffield United FC": "Sheffield United",

    # Southampton
    "Southampton": "Southampton",
    "Southampton FC": "Southampton",

    # Sunderland
    "Sunderland": "Sunderland",
    "Sunderland AFC": "Sunderland",

    # Tottenham
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "Spurs": "Tottenham",

    # Watford
    "Watford": "Watford",
    "Watford FC": "Watford",

    # West Brom
    "West Brom": "West Brom",
    "West Bromwich Albion": "West Brom",
    "West Bromwich Albion FC": "West Brom",

    # West Ham
    "West Ham": "West Ham",
    "West Ham United": "West Ham",
    "West Ham United FC": "West Ham",

    # Wolverhampton
    "Wolves": "Wolves",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton Wanderers FC": "Wolves",
}


# =====================================================
# TEAM NAME NORMALIZATION
# =====================================================

def clean_team_name(name):
    """
    Normalize capitalization, punctuation,
    whitespace and special characters.
    """

    name = unicodedata.normalize(
        "NFKD",
        str(name)
    )

    name = name.encode(
        "ascii",
        "ignore"
    ).decode("ascii")

    name = name.casefold().strip()

    name = name.replace("&", "and")

    name = re.sub(
        r"[^a-z0-9]+",
        " ",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    return name


def normalize_team_name(name):
    """
    Convert API names into historical CSV names.

    Unknown names are reported rather than
    automatically matched to a different club.
    """

    cleaned = clean_team_name(name)

    lookup = {
        clean_team_name(alias): canonical
        for alias, canonical in TEAM_ALIASES.items()
    }

    if cleaned in lookup:
        return lookup[cleaned]

    # Attempt to remove common suffixes
    suffixes = [
        " fc",
        " afc",
        " football club",
    ]

    for suffix in suffixes:

        if cleaned.endswith(suffix):

            shortened = cleaned[
                :-len(suffix)
            ].strip()

            if shortened in lookup:
                return lookup[shortened]

    suggestions = get_close_matches(
        cleaned,
        list(lookup.keys()),
        n=3,
        cutoff=0.65
    )

    possible_matches = list(dict.fromkeys(
        lookup[item]
        for item in suggestions
    ))

    raise ValueError(
        f"Unknown team name: {name}. "
        f"Possible matches: {possible_matches}"
    )


def get_api_team_name(team_data):
    """Prefer full API names over abbreviations."""

    if not isinstance(team_data, dict):
        return None

    return (
        team_data.get("name")
        or team_data.get("shortName")
    )


def validate_fixture_team_names(matches):
    """
    Check all API match records together.

    Reports all unknown names in one error,
    preventing repeated one-by-one failures.
    """

    unknown_names = set()

    for match in matches:

        for key in ["homeTeam", "awayTeam"]:

            team = match.get(key) or {}

            name = get_api_team_name(team)

            if not name:
                continue

            try:
                normalize_team_name(name)

            except ValueError:
                unknown_names.add(name)

    if unknown_names:

        unknown_text = "\n".join(
            sorted(unknown_names)
        )

        raise ValueError(
            "Unrecognized API team names:\n"
            f"{unknown_text}\n\n"
            "Add verified aliases to TEAM_ALIASES."
        )

    print(
        "All API fixture team names recognized!"
    )


# =====================================================
# CURRENT SEASON HISTORICAL RESULTS
# =====================================================

def fetch_match_records():
    """Get one consistent current-season API snapshot per process."""
    if not FOOTBALL_API_KEY:
        raise ValueError("FOOTBALL_API_KEY missing. Check your root .env file.")
    return _fetch_match_records()


from functools import lru_cache

@lru_cache(maxsize=1)
def _fetch_match_records():
    response = requests.get(
        f"{FOOTBALL_API_BASE_URL}/competitions/PL/matches",
        headers={"X-Auth-Token": FOOTBALL_API_KEY},
        params={"season": CURRENT_SEASON_START_YEAR}, timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    matches = response.json().get("matches", [])
    if len(matches) != 380 or len({m.get("id") for m in matches}) != 380 or any(m.get("id") is None for m in matches):
        raise ValueError("Expected 380 unique Premier League fixtures; refusing incomplete refresh.")
    return matches


def download_current_results():
    """Use official FINISHED scores from the same API as fixtures.

    Same-day completed games count in standings, but feature generation
    still excludes all matches on the fixture's calendar date.
    """
    rows = []
    for match in fetch_match_records():
        if match.get("status") != "FINISHED":
            continue
        score = (match.get("score") or {}).get("fullTime") or {}
        hg, ag = score.get("home"), score.get("away")
        if not isinstance(hg, int) or not isinstance(ag, int) or min(hg, ag) < 0:
            raise ValueError(f"Invalid finished score for match {match.get('id')}")
        kickoff = pd.to_datetime(match["utcDate"], utc=True)
        if kickoff > pd.Timestamp.now(tz="UTC"):
            raise ValueError("Finished match has a future kickoff")
        rows.append({"Date": pd.Timestamp(kickoff.date()),
                     "HomeTeam": normalize_team_name(get_api_team_name(match["homeTeam"])),
                     "AwayTeam": normalize_team_name(get_api_team_name(match["awayTeam"])),
                     "FTHG": hg, "FTAG": ag,
                     "FTR": "H" if hg > ag else "A" if hg < ag else "D",
                     "Season": CURRENT_SEASON})
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"])
    if df.empty or df.duplicated(["HomeTeam", "AwayTeam"]).any():
        raise ValueError("No completed matches or duplicate completed fixtures")
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = RAW_DATA_DIR / f"premier_league_{CURRENT_SEASON_START_YEAR}_{str(CURRENT_SEASON_START_YEAR+1)[-2:]}.csv"
    df.to_csv(output, index=False)
    print("Current-season completed matches:", len(df))
    return df


# =====================================================
# LOAD AND COMBINE HISTORICAL RESULTS
# =====================================================

def load_historical_results():
    """
    Combine the five historical seasons
    with completed current-season matches.
    """

    path = (
        RAW_DATA_DIR
        / "premier_league_raw.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Historical dataset missing. "
            "Run python -m src.data_collection."
        )

    historical = pd.read_csv(path)

    historical["Date"] = pd.to_datetime(
        historical["Date"]
    )

    columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "Season",
    ]

    historical = historical[
        columns
    ].copy()

    current = download_current_results()

    # Normalize team names across historical
    # and current-season datasets.
    for dataset in [historical, current]:

        dataset["HomeTeam"] = (
            dataset["HomeTeam"]
            .apply(normalize_team_name)
        )

        dataset["AwayTeam"] = (
            dataset["AwayTeam"]
            .apply(normalize_team_name)
        )

    print(
        "Historical and current-season "
        "team names standardized."
    )


    combined = pd.concat(
        [
            historical,
            current[columns]
        ],
        ignore_index=True
    )

    combined = combined.sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    combined = combined.drop_duplicates(
        subset=[
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ],
        keep="last"
    )

    print(
        "Total available historical matches:",
        len(combined)
    )

    return combined


# =====================================================
# FETCH UPCOMING PREMIER LEAGUE FIXTURES
# =====================================================

def fetch_upcoming_fixtures():
    """
    Fetch scheduled Premier League fixtures.

    Select the matchday of the earliest
    upcoming fixture.

    A matchday may contain fewer than ten
    remaining fixtures.
    """

    print("\nFetching upcoming fixtures...")
    matches = fetch_match_records()

    if not matches:
        raise ValueError(
            "API returned no Premier League matches."
        )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    upcoming = []

    for match in matches:

        status = match.get("status")

        if status not in [
            "SCHEDULED",
            "TIMED",
        ]:
            continue

        utc_date = match.get(
            "utcDate"
        )

        if not utc_date:
            continue

        kickoff = pd.to_datetime(
            utc_date,
            utc=True
        )

        if kickoff <= now:
            continue

        home_name = get_api_team_name(
            match.get("homeTeam") or {}
        )

        away_name = get_api_team_name(
            match.get("awayTeam") or {}
        )

        if not home_name or not away_name:
            continue

        upcoming.append({
            "MatchID": match.get("id"),
            "Matchday": match.get("matchday"),
            "KickoffUTC": kickoff,
            "OriginalHomeTeam": home_name,
            "OriginalAwayTeam": away_name,
        })

    if not upcoming:
        raise ValueError(
            "No future scheduled Premier League "
            "fixtures were found."
        )

    fixtures = pd.DataFrame(
        upcoming
    )

    fixtures = fixtures.sort_values(
        "KickoffUTC"
    ).reset_index(drop=True)

    first_fixture = fixtures.iloc[0]

    next_matchday = first_fixture[
        "Matchday"
    ]

    if pd.notna(next_matchday):

        fixtures = fixtures[
            fixtures["Matchday"]
            == next_matchday
        ].copy()

    else:

        first_kickoff = first_fixture[
            "KickoffUTC"
        ]

        fixtures = fixtures[
            fixtures["KickoffUTC"]
            <= (
                first_kickoff
                + pd.Timedelta(days=7)
            )
        ].copy()

    fixtures = fixtures.sort_values(
        "KickoffUTC"
    ).reset_index(drop=True)

    # Validate the selected fixtures together.
    selected_matches = []

    for _, fixture in fixtures.iterrows():

        selected_matches.append({
            "homeTeam": {
                "name": fixture["OriginalHomeTeam"]
            },
            "awayTeam": {
                "name": fixture["OriginalAwayTeam"]
            }
        })

    validate_fixture_team_names(
        selected_matches
    )

    fixtures["HomeTeam"] = (
        fixtures["OriginalHomeTeam"]
        .apply(normalize_team_name)
    )

    fixtures["AwayTeam"] = (
        fixtures["OriginalAwayTeam"]
        .apply(normalize_team_name)
    )

    fixtures = fixtures.drop(
        columns=[
            "OriginalHomeTeam",
            "OriginalAwayTeam",
        ]
    )

    print(
        "Upcoming fixtures found:",
        len(fixtures)
    )

    if pd.notna(next_matchday):
        print(
            "Selected matchday:",
            int(next_matchday)
        )

    return fixtures


# =====================================================
# BUILD TEAM MATCH HISTORIES
# =====================================================

def build_team_history(
    results,
    fixture_date
):
    """
    Build each team's historical match records.

    Only matches on dates strictly earlier
    than the target fixture are included.
    """

    eligible = results[
        results["Date"] < fixture_date
    ].sort_values("Date")

    histories = {}

    def add_record(team, record):

        if team not in histories:
            histories[team] = []

        histories[team].append(
            record
        )

    for _, match in eligible.iterrows():

        home = match["HomeTeam"]
        away = match["AwayTeam"]

        home_goals = int(
            match["FTHG"]
        )

        away_goals = int(
            match["FTAG"]
        )

        if home_goals > away_goals:
            home_points = 3
            away_points = 0

        elif home_goals < away_goals:
            home_points = 0
            away_points = 3

        else:
            home_points = 1
            away_points = 1

        add_record(
            home,
            {
                "date": match["Date"],
                "venue": "Home",
                "goals_scored": home_goals,
                "goals_conceded": away_goals,
                "points": home_points,
            }
        )

        add_record(
            away,
            {
                "date": match["Date"],
                "venue": "Away",
                "goals_scored": away_goals,
                "goals_conceded": home_goals,
                "points": away_points,
            }
        )

    return histories


# =====================================================
# GENERATE PRE-MATCH FEATURES
# =====================================================

def build_fixture_features(
    fixtures,
    historical_results
):
    """
    Generate the exact 38 features used
    during classification model training.
    """

    rows = []
    elo_cache = {}

    for _, fixture in fixtures.iterrows():

        kickoff = fixture[
            "KickoffUTC"
        ]

        fixture_date = pd.Timestamp(
            kickoff.date()
        )

        histories = build_team_history(
            historical_results,
            fixture_date
        )

        home = fixture["HomeTeam"]
        away = fixture["AwayTeam"]

        home_history = histories.get(
            home, []
        )

        away_history = histories.get(
            away, []
        )

        home_stats = get_team_statistics(
            home_history,
            fixture_date,
            venue="Home"
        )

        away_stats = get_team_statistics(
            away_history,
            fixture_date,
            venue="Away"
        )

        row = fixture.to_dict()

        for key in [
            "Last5Points",
            "AvgGoalsScored",
            "AvgGoalsConceded",
            "WinRate",
            "GoalDifference",
        ]:

            row[f"Home{key}"] = (
                home_stats[key]
            )

            row[f"Away{key}"] = (
                away_stats[key]
            )

        row["HomeHomeWinRate"] = (
            home_stats["VenueWinRate"]
        )

        row["AwayAwayWinRate"] = (
            away_stats["VenueWinRate"]
        )

        row["HomeRestDays"] = (
            home_stats["RestDays"]
        )

        row["AwayRestDays"] = (
            away_stats["RestDays"]
        )

        row["RestDaysDifference"] = (
            home_stats["RestDays"]
            - away_stats["RestDays"]
        )

        row["FormDifference"] = (
            home_stats["Last5Points"]
            - away_stats["Last5Points"]
        )

        row["GoalDifferenceDifference"] = (
            home_stats["GoalDifference"]
            - away_stats["GoalDifference"]
        )

        row["HomeHistoryMatches"] = len(
            home_history
        )

        row["AwayHistoryMatches"] = len(
            away_history
        )

        date_key = str(fixture_date.date())
        if date_key not in elo_cache:
            elo_cache[date_key] = elo_ratings_before(historical_results, fixture_date)
        ratings = elo_cache[date_key]
        add_advanced_fixture_features(
            row, home_history, away_history,
            ratings.get(home, 1500.0), ratings.get(away, 1500.0),
        )
        rows.append(row)

    result = pd.DataFrame(
        rows
    )

    missing_columns = [
        col
        for col in FEATURE_COLUMNS
        if col not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    if result[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError(
            "Generated features contain missing values."
        )

    print(
        f"Generated {len(FEATURE_COLUMNS)} "
        "features per fixture."
    )

    low_history = result[
        (result["HomeHistoryMatches"] < 5)
        | (result["AwayHistoryMatches"] < 5)
    ]

    if not low_history.empty:

        print(
            "Warning: Some fixtures involve teams "
            "with fewer than five historical matches."
        )

        print(
            low_history[
                [
                    "HomeTeam",
                    "AwayTeam",
                    "HomeHistoryMatches",
                    "AwayHistoryMatches",
                ]
            ].to_string(index=False)
        )

    return result


# =====================================================
# LOAD SAVED ML MODELS
# =====================================================

def load_models():
    """Load all previously trained models."""

    model_files = {
        "KNN": "knn.joblib",
        "SVM": "svm.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
    }

    classifiers = {}

    for name, filename in model_files.items():

        path = (
            MODELS_DIR / filename
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing model: {filename}"
            )

        classifiers[name] = joblib.load(
            path
        )

        print(
            f"Loaded classifier: {name}"
        )

    regression_path = (
        MODELS_DIR
        / "goal_regressor.joblib"
    )

    policy_path = (
        MODELS_DIR
        / "q_learning_policy.joblib"
    )

    if not regression_path.exists():
        raise FileNotFoundError(
            "Goal regressor missing."
        )

    if not policy_path.exists():
        raise FileNotFoundError(
            "Q-Learning policy missing."
        )

    regression = joblib.load(
        regression_path
    )

    policy = joblib.load(
        policy_path
    )

    print("Loaded goal regression model.")
    print("Loaded Q-Learning strategy policy.")

    return (
        classifiers,
        regression,
        policy
    )


# =====================================================
# GENERATE FIXTURE PREDICTIONS
# =====================================================

def predict_fixtures(
    feature_df
):
    """
    Predict:
    - Match result
    - Match result confidence
    - Total goals
    - Over/Under 2.5
    - Q-Learning selected model
    """

    (
        classifiers,
        regression,
        policy,
    ) = load_models()

    saved_columns_path = MODELS_DIR / "feature_columns.joblib"
    if not saved_columns_path.exists():
        raise FileNotFoundError("Missing feature_columns.joblib. Retrain models first.")
    saved_columns = joblib.load(saved_columns_path)
    if list(saved_columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            "Model feature schema mismatch. Run feature engineering, "
            "train_models, goal_regressor, and q_learning in that order."
        )
    X = feature_df[FEATURE_COLUMNS]

    predictions = (
        feature_df.copy()
    )

    # ---------------------------------------------
    # Classification predictions
    # ---------------------------------------------

    for name, model in classifiers.items():

        labels = model.predict(
            X
        )

        probabilities = model.predict_proba(
            X
        )

        classes = list(
            model.classes_
        )

        predictions[
            PREDICTION_COLUMNS[name]
        ] = labels

        predicted_indices = [
            classes.index(label)
            for label in labels
        ]

        confidence = probabilities[
            np.arange(len(labels)),
            predicted_indices
        ]

        predictions[
            CONFIDENCE_COLUMNS[name]
        ] = confidence

        safe_name = name.replace(
            " ", "_"
        )

        for label in ["H", "D", "A"]:

            column_name = (
                f"{safe_name}_P_{label}"
            )

            if label in classes:

                class_index = classes.index(
                    label
                )

                predictions[column_name] = (
                    probabilities[:, class_index]
                )

            else:

                predictions[column_name] = 0.0

    # ---------------------------------------------
    # Total goals regression
    # ---------------------------------------------

    goal_predictions = regression.predict(
        X
    )

    goal_predictions = np.maximum(
        goal_predictions,
        0.0
    )

    predictions["PredictedTotalGoals"] = (
        goal_predictions
    )

    # The binary model is independent of the total-goals point estimate.
    over_path = MODELS_DIR / "over_under_classifier.joblib"
    over_model = joblib.load(over_path)
    over_prob = over_model.predict_proba(X)[:, list(over_model.classes_).index(1)]
    if not np.isfinite(over_prob).all() or ((over_prob < 0) | (over_prob > 1)).any():
        raise ValueError("Invalid Over/Under probabilities")
    predictions["ProbOver2_5"] = over_prob
    predictions["ProbUnder2_5"] = 1 - over_prob
    predictions["OverUnder2_5"] = np.where(over_prob >= 0.5, "Over", "Under")
    predictions["PredictionGeneratedUTC"] = pd.Timestamp.now(tz="UTC").isoformat()

    # ---------------------------------------------
    # Q-Learning Strategy Maker
    # ---------------------------------------------

    q_table = policy[
        "q_table"
    ]

    fallback_model = policy.get(
        "default_action",
        "KNN"
    )

    selected_models = []
    selected_labels = []
    selected_confidences = []
    learned_q_values = []
    explanations = []

    for _, row in predictions.iterrows():

        state = create_state(
            row
        )

        values = q_table.get(
            state
        )

        if values is None:

            selected_action = (
                fallback_model
            )

            q_value = np.nan

            explanation = (
                "Previously unseen state. "
                f"Selected fallback model: "
                f"{fallback_model}."
            )

        else:

            best_index = int(
                np.argmax(values)
            )

            selected_action = ACTIONS[
                best_index
            ]

            q_value = float(
                values[best_index]
            )

            explanation = (
                f"State {state}: Q-Learning selected "
                f"{selected_action} based on its "
                "estimated historical reward."
            )

        selected_models.append(
            selected_action
        )

        selected_labels.append(
            row[
                PREDICTION_COLUMNS[
                    selected_action
                ]
            ]
        )

        selected_confidences.append(
            float(
                row[
                    CONFIDENCE_COLUMNS[
                        selected_action
                    ]
                ]
            )
        )

        learned_q_values.append(
            q_value
        )

        explanations.append(
            explanation
        )

    predictions["SelectedModel"] = (
        selected_models
    )

    predictions["PredictedResult"] = (
        selected_labels
    )

    predictions["SelectedConfidence"] = (
        selected_confidences
    )

    predictions["LearnedQValue"] = (
        learned_q_values
    )

    predictions["StrategyExplanation"] = (
        explanations
    )

    predictions["ResultLabel"] = (
        predictions["PredictedResult"]
        .map(RESULT_LABELS)
    )

    # ---------------------------------------------
    # Display-friendly values
    # ---------------------------------------------

    predictions["KickoffUTC"] = (
        predictions["KickoffUTC"]
        .astype(str)
    )

    return predictions


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():

    print(
        "\n===== WEEKLY PREMIER LEAGUE PREDICTOR ====="
    )

    # Step 1: Historical football results
    historical_results = (
        load_historical_results()
    )

    # Step 2: Upcoming fixtures
    fixtures = (
        fetch_upcoming_fixtures()
    )

    # Step 3: Feature engineering
    featured_fixtures = build_fixture_features(
        fixtures,
        historical_results
    )

    # Step 4: ML and Q-Learning predictions
    predictions = predict_fixtures(
        featured_fixtures
    )

    # Step 5: Save predictions
    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PREDICTIONS_DIR
        / "weekly_predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False
    )

    display_columns = [
        "HomeTeam",
        "AwayTeam",
        "ResultLabel",
        "PredictedTotalGoals",
        "OverUnder2_5",
        "ProbOver2_5",
        "SelectedModel",
    ]

    print(
        "\n===== MATCH PREDICTIONS ====="
    )

    print(
        predictions[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}"
        )
    )

    print(
        "\nPredictions saved to:",
        output_path
    )

    print(
        "Total fixtures predicted:",
        len(predictions)
    )

    print(
        "\n===== PREDICTION ENGINE COMPLETED ====="
    )

    return predictions


# =====================================================
# RUN PROGRAM
# =====================================================

if __name__ == "__main__":
    main()

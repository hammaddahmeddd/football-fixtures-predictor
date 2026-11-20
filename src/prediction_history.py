
"""
Football Fixtures Result Predictor V2
Prediction History and Live Accuracy Tracking

Run:
    python -m src.prediction_history
"""

import pandas as pd
import numpy as np

from config import PREDICTIONS_DIR, OUTPUTS_DIR

from src.predictions import fetch_match_records


HISTORY_PATH = (
    PREDICTIONS_DIR / "prediction_history.csv"
)

WEEKLY_PATH = (
    PREDICTIONS_DIR / "weekly_predictions.csv"
)

METRICS_PATH = (
    OUTPUTS_DIR / "metrics" /
    "live_accuracy_summary.csv"
)

PREDICTION_COLUMNS = [
    "MatchID",
    "Matchday",
    "KickoffUTC",
    "HomeTeam",
    "AwayTeam",
    "PredictionGeneratedUTC",
    "PredictedResult",
    "SelectedModel",
    "SelectedConfidence",
    "PredictedTotalGoals",
    "OverUnder2_5",
    "ProbOver2_5",
]

RESULT_COLUMNS = [
    "Status",
    "ActualHomeGoals",
    "ActualAwayGoals",
    "ActualResult",
    "ActualTotalGoals",
    "ActualOverUnder",
    "ResultCorrect",
    "OverUnderCorrect",
    "AbsoluteGoalError",
    "ResultRegisteredUTC",
]

ALL_COLUMNS = (
    PREDICTION_COLUMNS + RESULT_COLUMNS
)


def load_history():

    if HISTORY_PATH.exists():

        history = pd.read_csv(
            HISTORY_PATH,
            dtype={"MatchID": "string"}
        )

        if not set(ALL_COLUMNS).issubset(
            history.columns
        ):
            raise ValueError(
                "Invalid prediction history schema."
            )

        if history["MatchID"].duplicated().any():
            raise ValueError(
                "Duplicate predictions in history."
            )

        return history

    return pd.DataFrame(
        columns=ALL_COLUMNS
    )


def load_weekly_predictions():

    if not WEEKLY_PATH.exists():

        raise FileNotFoundError(
            "Weekly predictions missing. "
            "Run python -m src.predictions first."
        )

    weekly = pd.read_csv(
        WEEKLY_PATH,
        dtype={"MatchID": "string"}
    )

    missing = set(
        PREDICTION_COLUMNS
    ) - set(weekly.columns)

    if missing:

        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    return weekly


def register_predictions(history, weekly, matches):

    existing_ids = set(
        history["MatchID"].astype(str)
    )

    records = []

    now = pd.Timestamp.now(tz="UTC")

    for _, row in weekly.iterrows():

        match_id = str(row["MatchID"])

        if match_id in existing_ids:
            continue

        match = matches.get(match_id)

        if match is None:
            continue

        if match["status"] not in [
            "SCHEDULED", "TIMED"
        ]:
            continue

        kickoff = pd.to_datetime(
            match["utcDate"],
            utc=True
        )

        generated = pd.to_datetime(
            row["PredictionGeneratedUTC"],
            utc=True
        )

        # Never register predictions after kickoff.
        if not generated <= now < kickoff:
            continue

        record = {
            col: row[col]
            for col in PREDICTION_COLUMNS
        }

        record["MatchID"] = match_id

        for col in RESULT_COLUMNS:
            record[col] = None

        record["Status"] = "Pending"

        records.append(record)

        existing_ids.add(match_id)

    if records:

        history = pd.concat(
            [
                history,
                pd.DataFrame(records)
            ],
            ignore_index=True
        )

    print(
        "New predictions registered:",
        len(records)
    )

    return history


def update_actual_results(history, matches):

    updated = 0

    for index, row in history.iterrows():

        match_id = str(row["MatchID"])

        match = matches.get(match_id)

        if match is None:
            continue

        if match["status"] != "FINISHED":
            continue

        actual_kickoff = pd.to_datetime(
            match["utcDate"],
            utc=True
        )

        generated = pd.to_datetime(
            row["PredictionGeneratedUTC"],
            utc=True
        )

        # Protect against rescheduled kickoff times.
        if generated >= actual_kickoff:

            history.at[index, "Status"] = (
                "Invalid"
            )

            continue

        score = (
            match.get("score") or {}
        ).get("fullTime") or {}

        home_goals = score.get("home")
        away_goals = score.get("away")

        if (
            not isinstance(home_goals, int)
            or not isinstance(away_goals, int)
            or min(home_goals, away_goals) < 0
        ):

            raise ValueError(
                f"Invalid score for match {match_id}"
            )

        total_goals = home_goals + away_goals

        if home_goals > away_goals:
            actual_result = "H"

        elif home_goals < away_goals:
            actual_result = "A"

        else:
            actual_result = "D"

        actual_over_under = (
            "Over"
            if total_goals >= 3
            else "Under"
        )

        result_correct = int(
            row["PredictedResult"]
            == actual_result
        )

        over_under_correct = int(
            row["OverUnder2_5"]
            == actual_over_under
        )

        absolute_goal_error = abs(
            float(row["PredictedTotalGoals"])
            - total_goals
        )

        history.at[index, "Status"] = (
            "Finished"
        )

        history.at[index, "ActualHomeGoals"] = (
            home_goals
        )

        history.at[index, "ActualAwayGoals"] = (
            away_goals
        )

        history.at[index, "ActualResult"] = (
            actual_result
        )

        history.at[index, "ActualTotalGoals"] = (
            total_goals
        )

        history.at[index, "ActualOverUnder"] = (
            actual_over_under
        )

        history.at[index, "ResultCorrect"] = (
            result_correct
        )

        history.at[index, "OverUnderCorrect"] = (
            over_under_correct
        )

        history.at[index, "AbsoluteGoalError"] = (
            absolute_goal_error
        )

        if pd.isna(row["ResultRegisteredUTC"]):

            history.at[
                index,
                "ResultRegisteredUTC"
            ] = pd.Timestamp.now(
                tz="UTC"
            ).isoformat()

            updated += 1

    print(
        "New completed results registered:",
        updated
    )

    return history


def calculate_live_accuracy(history):

    finished = history[
        history["Status"] == "Finished"
    ].copy()

    pending = history[
        history["Status"] == "Pending"
    ]

    total = len(finished)

    correct_results = pd.to_numeric(
        finished["ResultCorrect"]
    ).sum()

    correct_over_under = pd.to_numeric(
        finished["OverUnderCorrect"]
    ).sum()

    result_accuracy = (
        correct_results / total
        if total else np.nan
    )

    over_under_accuracy = (
        correct_over_under / total
        if total else np.nan
    )

    goal_mae = (
        pd.to_numeric(
            finished["AbsoluteGoalError"]
        ).mean()
        if total else np.nan
    )

    summary = pd.DataFrame([{
        "TotalPredictions": len(history),
        "CompletedMatches": total,
        "PendingMatches": len(pending),
        "CorrectResultPredictions": int(
            correct_results
        ),
        "ResultAccuracy": result_accuracy,
        "CorrectOverUnderPredictions": int(
            correct_over_under
        ),
        "OverUnderAccuracy": over_under_accuracy,
        "GoalMAE": goal_mae,
    }])

    return summary


def main():

    print(
        "\n===== LIVE PREDICTION TRACKER ====="
    )

    history = load_history()

    weekly = load_weekly_predictions()

    matches = {
        str(match["id"]): match
        for match in fetch_match_records()
    }

    history = register_predictions(
        history,
        weekly,
        matches
    )

    history = update_actual_results(
        history,
        matches
    )

    summary = calculate_live_accuracy(
        history
    )

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history = history.sort_values(
        ["KickoffUTC", "MatchID"]
    ).reset_index(drop=True)

    history.to_csv(
        HISTORY_PATH,
        index=False
    )

    summary.to_csv(
        METRICS_PATH,
        index=False
    )

    print(
        "\n===== LIVE ACCURACY SUMMARY ====="
    )

    print(
        summary.to_string(index=False)
    )

    print(
        "\nPrediction history saved:",
        HISTORY_PATH
    )

    print(
        "Live accuracy saved:",
        METRICS_PATH
    )

    print(
        "\n===== TRACKING COMPLETED ====="
    )


if __name__ == "__main__":
    main()

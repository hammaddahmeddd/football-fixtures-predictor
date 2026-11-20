
"""
Football Fixtures Result Predictor
Step 10: Full-Season Monte Carlo Simulation

Run:
    python -m src.season_simulation

Uses:
    - Completed Premier League match results
    - Remaining fixtures from football-data.org
    - Four classification models
    - Q-Learning strategy selector
    - Goal regression model

Produces:
    - Projected final standings
    - Title probabilities
    - Top-four probabilities
    - Fifth-place probabilities
    - Relegation probabilities
    - Charts and CSV exports

Method:
    Monte Carlo simulation with fixed pre-match
    probability estimates for future fixtures.

Important limitation:
    Model probabilities are not yet proven to
    be well-calibrated. Simulation results are
    estimates, not guaranteed outcomes.
"""

from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from config import (
    FOOTBALL_API_KEY,
    FOOTBALL_API_BASE_URL,
    CURRENT_SEASON_START_YEAR,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
)

from src.predictions import (
    fetch_match_records,
    get_api_team_name,
    normalize_team_name,
    load_historical_results,
    build_fixture_features,
    predict_fixtures,
)


# ===============================================
# SETTINGS
# ===============================================

SIMULATIONS = 1000
RANDOM_SEED = 42

EXPECTED_TEAMS = 20
EXPECTED_MATCHES = 380
EXPECTED_MATCHES_PER_TEAM = 38

GRAPH_DIR = OUTPUTS_DIR / "graphs"
METRICS_DIR = OUTPUTS_DIR / "metrics"

for folder in [
    GRAPH_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ===============================================
# FETCH THE ENTIRE SEASON SCHEDULE
# ===============================================

def fetch_season_schedule():
    """
    Download the entire current season schedule.

    Unlike weekly predictions, this function
    retrieves every remaining fixture.
    """

    matches = fetch_match_records()

    if not matches:
        raise ValueError(
            "API returned no season fixtures."
        )

    records = []

    for match in matches:

        home = get_api_team_name(
            match.get("homeTeam") or {}
        )

        away = get_api_team_name(
            match.get("awayTeam") or {}
        )

        # Skip incomplete placeholder entries
        if not home or not away:
            continue

        kickoff = match.get("utcDate")

        if not kickoff:
            continue

        records.append({
            "MatchID": match.get("id"),
            "Matchday": match.get("matchday"),
            "KickoffUTC": pd.to_datetime(
                kickoff,
                utc=True
            ),
            "HomeTeam": normalize_team_name(
                home
            ),
            "AwayTeam": normalize_team_name(
                away
            ),
            "Status": match.get("status"),
        })

    schedule = pd.DataFrame(records)

    if schedule.empty:
        raise ValueError(
            "No valid season fixtures found."
        )

    schedule = schedule.sort_values(
        "KickoffUTC"
    ).reset_index(drop=True)

    print(
        "Total API schedule entries:",
        len(schedule)
    )

    return schedule


# ===============================================
# LOAD COMPLETED MATCHES
# ===============================================

def load_completed_matches():
    """
    Load actual completed results for the
    current Premier League season.
    """

    history = load_historical_results()

    current = history[
        history["Season"].astype(str)
        == (
            f"{CURRENT_SEASON_START_YEAR}-"
            f"{str(CURRENT_SEASON_START_YEAR + 1)[-2:]}"
        )
    ].copy()

    if current.empty:
        raise ValueError(
            "No completed current-season matches."
        )

    current["Date"] = pd.to_datetime(
        current["Date"]
    )

    current = current.drop_duplicates(
        subset=[
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
    ).reset_index(drop=True)

    print(
        "Completed season matches:",
        len(current)
    )

    return history, current


# ===============================================
# IDENTIFY REMAINING FIXTURES
# ===============================================

def get_remaining_fixtures(
    schedule,
    completed,
):
    """
    Separate completed matches from upcoming
    matches and validate the season schedule.

    Fail rather than silently simulating an
    incomplete Premier League season.
    """

    now = pd.Timestamp.now(tz="UTC")

    completed_pairs = set(
        zip(
            completed["HomeTeam"],
            completed["AwayTeam"]
        )
    )

    remaining_rows = []

    for _, match in schedule.iterrows():

        pair = (
            match["HomeTeam"],
            match["AwayTeam"]
        )

        # Actual completed results take priority
        if pair in completed_pairs:
            continue

        if match["Status"] not in [
            "SCHEDULED",
            "TIMED",
            "POSTPONED",
        ]:
            raise ValueError(
                "Unresolved fixture status: "
                f"{match['HomeTeam']} vs "
                f"{match['AwayTeam']} "
                f"({match['Status']})"
            )

        if match["KickoffUTC"] <= now:
            raise ValueError(
                "An uncompleted fixture has a "
                "past scheduled kickoff: "
                f"{match['HomeTeam']} vs "
                f"{match['AwayTeam']}. "
                "Refresh the match data."
            )

        remaining_rows.append(
            match.to_dict()
        )

    remaining = pd.DataFrame(
        remaining_rows
    )

    if remaining.empty:
        raise ValueError(
            "No remaining fixtures found."
        )

    remaining = remaining.sort_values(
        "KickoffUTC"
    ).reset_index(drop=True)

    # Confirm all 20 current-season teams.
    teams = sorted(
        set(schedule["HomeTeam"])
        | set(schedule["AwayTeam"])
    )

    if len(teams) != EXPECTED_TEAMS:
        raise ValueError(
            f"Expected 20 teams, found {len(teams)}. "
            f"Teams: {teams}"
        )

    total = len(completed) + len(remaining)

    if total != EXPECTED_MATCHES:
        raise ValueError(
            "Incomplete or inconsistent schedule. "
            f"Completed={len(completed)}, "
            f"Remaining={len(remaining)}, "
            f"Total={total}, expected=380. "
            "Check API statuses and CSV coverage."
        )

    # Check that every team has exactly 38
    # completed + remaining matches.
    appearances = {
        team: 0 for team in teams
    }

    for dataset in [completed, remaining]:
        for _, row in dataset.iterrows():
            appearances[row["HomeTeam"]] += 1
            appearances[row["AwayTeam"]] += 1

    invalid = {
        team: count
        for team, count in appearances.items()
        if count != EXPECTED_MATCHES_PER_TEAM
    }

    if invalid:
        raise ValueError(
            "Incorrect matches per team: "
            f"{invalid}"
        )

    print(
        "Remaining matches:",
        len(remaining)
    )

    print(
        "Full 380-match schedule validated."
    )

    return remaining, teams


# ===============================================
# INITIAL LEAGUE TABLE
# ===============================================

def create_initial_table(completed, teams):
    """
    Build the real current standings from
    completed results.

    Each team starts the simulations with
    its actual points and goal difference.
    """

    table = pd.DataFrame({
        "Team": teams,
        "P": 0,
        "W": 0,
        "D": 0,
        "L": 0,
        "GF": 0,
        "GA": 0,
        "GD": 0,
        "Pts": 0,
    })

    table = table.set_index("Team")

    for _, match in completed.iterrows():

        home = match["HomeTeam"]
        away = match["AwayTeam"]

        hg = int(match["FTHG"])
        ag = int(match["FTAG"])

        table.loc[home, "P"] += 1
        table.loc[away, "P"] += 1

        table.loc[home, "GF"] += hg
        table.loc[home, "GA"] += ag

        table.loc[away, "GF"] += ag
        table.loc[away, "GA"] += hg

        if hg > ag:
            table.loc[home, "W"] += 1
            table.loc[away, "L"] += 1
            table.loc[home, "Pts"] += 3

        elif hg < ag:
            table.loc[away, "W"] += 1
            table.loc[home, "L"] += 1
            table.loc[away, "Pts"] += 3

        else:
            table.loc[home, "D"] += 1
            table.loc[away, "D"] += 1
            table.loc[home, "Pts"] += 1
            table.loc[away, "Pts"] += 1

    table["GD"] = (
        table["GF"] - table["GA"]
    )

    return table


# ===============================================
# GET FUTURE MATCH PROBABILITIES
# ===============================================

def prepare_future_predictions(
    remaining,
    history,
):
    """
    Generate predictions for all remaining
    fixtures using existing trained models.
    """

    fixtures = remaining[
        [
            "MatchID",
            "Matchday",
            "KickoffUTC",
            "HomeTeam",
            "AwayTeam",
        ]
    ].copy()

    featured = build_fixture_features(
        fixtures,
        history
    )

    predictions = predict_fixtures(
        featured
    )

    # Extract predictive probabilities from the selected classifier.
    # Uncalibrated probabilities must be treated as experimental, not
    # as verified real-world likelihoods.
    probabilities = []

    for _, row in predictions.iterrows():

        selected = row["SelectedModel"]

        prefix = selected.replace(
            " ", "_"
        )

        values = np.array([
            row[f"{prefix}_P_H"],
            row[f"{prefix}_P_D"],
            row[f"{prefix}_P_A"],
        ], dtype=float)

        # Protect against invalid probability
        # output and normalize minor rounding.
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or values.sum() <= 0
        ):
            raise ValueError(
                "Invalid model probabilities "
                f"for {row['HomeTeam']} vs "
                f"{row['AwayTeam']}"
            )

        values = values / values.sum()

        probabilities.append(values)

    probabilities = np.asarray(
        probabilities
    )

    predictions["ProbHome"] = probabilities[:, 0]
    predictions["ProbDraw"] = probabilities[:, 1]
    predictions["ProbAway"] = probabilities[:, 2]

    # Simple attacking/defensive relative
    # strength, scaled to the regression
    # model's expected total goals.
    home_strength = (
        predictions["HomeAvgGoalsScored"].to_numpy()
        + predictions["AwayAvgGoalsConceded"].to_numpy()
        + 0.5
    )

    away_strength = (
        predictions["AwayAvgGoalsScored"].to_numpy()
        + predictions["HomeAvgGoalsConceded"].to_numpy()
        + 0.5
    )

    total_strength = (
        home_strength + away_strength
    )

    expected_total = (
        predictions["PredictedTotalGoals"]
        .to_numpy(dtype=float)
    )

    expected_total = np.clip(
        expected_total, 0.5, 7.0
    )

    predictions["ExpectedHomeGoals"] = (
        expected_total
        * home_strength / total_strength
    )

    predictions["ExpectedAwayGoals"] = (
        expected_total
        * away_strength / total_strength
    )

    return predictions


# ===============================================
# SAMPLE SCORES GIVEN MATCH OUTCOME
# ===============================================

def sample_scorelines(
    result_codes,
    home_lambda,
    away_lambda,
    rng,
):
    """
    Generate plausible scores conditioned
    on the already sampled match outcome.

    Result code:
        0 = Home win
        1 = Draw
        2 = Away win

    Uses bounded rejection sampling,
    with a simple fallback for rare failures.
    """

    count = len(result_codes)

    hg = np.zeros(count, dtype=int)
    ag = np.zeros(count, dtype=int)

    unresolved = np.ones(
        count, dtype=bool
    )

    for _ in range(30):

        ids = np.flatnonzero(
            unresolved
        )

        if len(ids) == 0:
            break

        candidate_h = rng.poisson(
            home_lambda,
            len(ids)
        )

        candidate_a = rng.poisson(
            away_lambda,
            len(ids)
        )

        codes = result_codes[ids]

        valid = (
            ((codes == 0) & (candidate_h > candidate_a))
            | ((codes == 1) & (candidate_h == candidate_a))
            | ((codes == 2) & (candidate_h < candidate_a))
        )

        accepted = ids[valid]

        hg[accepted] = candidate_h[valid]
        ag[accepted] = candidate_a[valid]

        unresolved[accepted] = False

    # Fallback preserves the selected outcome.
    remaining = np.flatnonzero(
        unresolved
    )

    for index in remaining:

        code = result_codes[index]

        if code == 0:
            hg[index], ag[index] = 2, 1
        elif code == 1:
            hg[index], ag[index] = 1, 1
        else:
            hg[index], ag[index] = 1, 2

    return hg, ag


# ===============================================
# MONTE CARLO SEASON SIMULATION
# ===============================================

def run_monte_carlo(
    initial_table,
    predictions,
    teams,
):
    """
    Simulate every remaining fixture in
    SIMULATIONS independent possible seasons.

    Stores final points, scores and rankings.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    team_index = {
        team: index
        for index, team in enumerate(teams)
    }

    n_teams = len(teams)

    columns = [
        "P", "W", "D", "L",
        "GF", "GA", "Pts",
    ]

    # Shape: simulations x teams
    stats = {
        column: np.tile(
            initial_table.loc[
                teams, column
            ].to_numpy(dtype=int),
            (SIMULATIONS, 1)
        )
        for column in columns
    }

    all_runs = np.arange(
        SIMULATIONS
    )

    print(
        f"\nRunning {SIMULATIONS} "
        "Monte Carlo simulations..."
    )

    for _, match in predictions.iterrows():

        h = team_index[
            match["HomeTeam"]
        ]

        a = team_index[
            match["AwayTeam"]
        ]

        probs = np.array([
            match["ProbHome"],
            match["ProbDraw"],
            match["ProbAway"],
        ], dtype=float)

        outcomes = rng.choice(
            3,
            size=SIMULATIONS,
            p=probs
        )

        hg, ag = sample_scorelines(
            outcomes,
            float(match["ExpectedHomeGoals"]),
            float(match["ExpectedAwayGoals"]),
            rng,
        )

        home_win = (
            outcomes == 0
        ).astype(int)

        draw = (
            outcomes == 1
        ).astype(int)

        away_win = (
            outcomes == 2
        ).astype(int)

        stats["P"][:, h] += 1
        stats["P"][:, a] += 1

        stats["W"][:, h] += home_win
        stats["W"][:, a] += away_win

        stats["D"][:, h] += draw
        stats["D"][:, a] += draw

        stats["L"][:, h] += away_win
        stats["L"][:, a] += home_win

        stats["GF"][:, h] += hg
        stats["GA"][:, h] += ag

        stats["GF"][:, a] += ag
        stats["GA"][:, a] += hg

        stats["Pts"][:, h] += (
            3 * home_win + draw
        )

        stats["Pts"][:, a] += (
            3 * away_win + draw
        )

    goal_difference = (
        stats["GF"] - stats["GA"]
    )

    # Rank each simulated season.
    # Approximate tie-breakers:
    # points, goal difference, goals scored,
    # then alphabetical order.
    positions = np.zeros(
        (SIMULATIONS, n_teams),
        dtype=int
    )

    for simulation in all_runs:

        ranking = sorted(
            range(n_teams),
            key=lambda i: (
                -stats["Pts"][simulation, i],
                -goal_difference[simulation, i],
                -stats["GF"][simulation, i],
                teams[i],
            )
        )

        for position, team_id in enumerate(
            ranking, start=1
        ):
            positions[
                simulation, team_id
            ] = position

    summary = pd.DataFrame({
        "Team": teams,
        "ExpectedPoints": (
            stats["Pts"].mean(axis=0)
        ),
        "ExpectedWins": (
            stats["W"].mean(axis=0)
        ),
        "ExpectedDraws": (
            stats["D"].mean(axis=0)
        ),
        "ExpectedLosses": (
            stats["L"].mean(axis=0)
        ),
        "ExpectedGoalsFor": (
            stats["GF"].mean(axis=0)
        ),
        "ExpectedGoalsAgainst": (
            stats["GA"].mean(axis=0)
        ),
        "ExpectedGoalDifference": (
            goal_difference.mean(axis=0)
        ),
        "AveragePosition": (
            positions.mean(axis=0)
        ),
        "TitleProbability": (
            (positions == 1).mean(axis=0)
        ),
        "Top4Probability": (
            (positions <= 4).mean(axis=0)
        ),
        "FifthPlaceProbability": (
            (positions == 5).mean(axis=0)
        ),
        "Top6Probability": (
            (positions <= 6).mean(axis=0)
        ),
        "RelegationProbability": (
            (positions >= 18).mean(axis=0)
        ),
        "PointsP05": np.percentile(
            stats["Pts"], 5, axis=0
        ),
        "PointsP95": np.percentile(
            stats["Pts"], 95, axis=0
        ),
    })

    summary = summary.sort_values(
        [
            "ExpectedPoints",
            "ExpectedGoalDifference",
            "ExpectedGoalsFor",
        ],
        ascending=False
    ).reset_index(drop=True)

    summary.insert(
        0,
        "ProjectedPosition",
        np.arange(1, n_teams + 1)
    )

    # Export all simulated final points.
    points_df = pd.DataFrame(
        stats["Pts"],
        columns=teams
    )

    points_df.insert(
        0,
        "Simulation",
        np.arange(1, SIMULATIONS + 1)
    )

    # Invariant: all 380 fixtures must be
    # accounted for, 38 per team.
    if not np.all(
        stats["P"] == EXPECTED_MATCHES_PER_TEAM
    ):
        raise RuntimeError(
            "Simulated match counts are inconsistent."
        )

    return summary, points_df


# ===============================================
# SAVE RESULTS
# ===============================================

def save_outputs(
    summary,
    points_df,
    predictions,
):
    """Export league projections and fixtures."""

    summary.to_csv(
        METRICS_DIR
        / "season_simulation_summary.csv",
        index=False
    )

    probability_columns = [
        "Team",
        "TitleProbability",
        "Top4Probability",
        "FifthPlaceProbability",
        "Top6Probability",
        "RelegationProbability",
    ]

    summary[
        probability_columns
    ].to_csv(
        METRICS_DIR
        / "season_probabilities.csv",
        index=False
    )

    points_df.to_csv(
        METRICS_DIR
        / "simulated_season_points.csv",
        index=False
    )

    predictions.to_csv(
        PREDICTIONS_DIR
        / "remaining_fixtures_predictions.csv",
        index=False
    )

    print("\nSeason CSV files saved.")


# ===============================================
# VISUALIZATIONS
# ===============================================

def save_graphs(summary):
    """Create figures for dashboard and report."""

    # Expected final league points
    ordered = summary.sort_values(
        "ExpectedPoints"
    )

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    ax.barh(
        ordered["Team"],
        ordered["ExpectedPoints"]
    )

    ax.set_xlabel(
        "Expected Final Points"
    )

    ax.set_title(
        "Projected Premier League Final Points"
    )

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR
        / "projected_final_points.png",
        dpi=150
    )

    plt.close(fig)

    # Top-four probability
    ordered = summary.sort_values(
        "Top4Probability"
    )

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    ax.barh(
        ordered["Team"],
        ordered["Top4Probability"] * 100
    )

    ax.set_xlabel(
        "Top-Four Probability (%)"
    )

    ax.set_title(
        "Premier League Top-Four Probability"
    )

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR
        / "top4_probabilities.png",
        dpi=150
    )

    plt.close(fig)

    # Relegation probability
    ordered = summary.sort_values(
        "RelegationProbability"
    )

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    ax.barh(
        ordered["Team"],
        ordered["RelegationProbability"] * 100
    )

    ax.set_xlabel(
        "Relegation Probability (%)"
    )

    ax.set_title(
        "Premier League Relegation Probability"
    )

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR
        / "relegation_probabilities.png",
        dpi=150
    )

    plt.close(fig)

    print(
        "Season simulation graphs saved."
    )


# ===============================================
# MAIN
# ===============================================

def main():

    print(
        "\n===== FULL-SEASON PREMIER LEAGUE SIMULATION ====="
    )

    history, completed = (
        load_completed_matches()
    )

    schedule = (
        fetch_season_schedule()
    )

    remaining, teams = get_remaining_fixtures(
        schedule,
        completed
    )

    initial_table = create_initial_table(
        completed,
        teams
    )

    future_predictions = (
        prepare_future_predictions(
            remaining,
            history
        )
    )

    summary, points_df = run_monte_carlo(
        initial_table,
        future_predictions,
        teams
    )

    save_outputs(
        summary,
        points_df,
        future_predictions
    )

    save_graphs(
        summary
    )

    display = summary[
        [
            "ProjectedPosition",
            "Team",
            "ExpectedPoints",
            "TitleProbability",
            "Top4Probability",
            "RelegationProbability",
        ]
    ].copy()

    for column in [
        "TitleProbability",
        "Top4Probability",
        "RelegationProbability",
    ]:
        display[column] *= 100

    print(
        "\n===== PROJECTED FINAL LEAGUE TABLE ====="
    )

    print(
        display.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}"
        )
    )

    print(
        "\n===== SEASON SIMULATION COMPLETED ====="
    )

    return summary


if __name__ == "__main__":
    main()

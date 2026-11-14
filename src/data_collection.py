
from io import StringIO

import pandas as pd
import requests

from config import (
    FOOTBALL_DATA_URL,
    LEAGUE_CODE,
    RAW_DATA_DIR,
    SEASONS,
)

# Columns required for ML training
REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]

# Additional useful historical statistics
OPTIONAL_COLUMNS = [
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]


def season_to_code(season):
    """
    Convert '2024-25' into '2425'.
    """
    start_year, end_year = season.split("-")

    return start_year[-2:] + end_year[-2:]


def download_season(season):
    """
    Download Premier League data for one season.
    """

    season_code = season_to_code(season)

    url = (
        f"{FOOTBALL_DATA_URL}/"
        f"{season_code}/{LEAGUE_CODE}.csv"
    )

    print(f"\nDownloading season: {season}")
    print(f"Source: {url}")

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    dataframe = pd.read_csv(
        StringIO(response.text),
        low_memory=False
    )

    # Validate required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep required and available optional columns
    selected_columns = [
        column
        for column in (
            REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        )
        if column in dataframe.columns
    ]

    dataframe = dataframe[selected_columns].copy()

    # Add season information
    dataframe["Season"] = season

    print(
        f"Successfully downloaded "
        f"{len(dataframe)} matches"
    )

    return dataframe


def collect_data():
    """
    Download, combine and save all seasons.
    """

    all_seasons = []
    failed_seasons = []

    for season in SEASONS:
        try:
            season_data = download_season(season)
            all_seasons.append(season_data)

        except (
            requests.RequestException,
            ValueError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as error:
            print(f"Error downloading {season}: {error}")
            failed_seasons.append(season)

    if not all_seasons:
        raise RuntimeError(
            "No football datasets were downloaded."
        )

    # Combine all seasons
    combined_data = pd.concat(
        all_seasons,
        ignore_index=True
    )

    # Convert match dates
    combined_data["Date"] = pd.to_datetime(
        combined_data["Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Remove invalid dates and missing teams
    combined_data = combined_data.dropna(
        subset=[
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
        ]
    )

    # Arrange matches chronologically
    combined_data = combined_data.sort_values(
        ["Date", "Season", "HomeTeam"],
        kind="stable"
    ).reset_index(drop=True)

    # Save historical dataset
    output_path = (
        RAW_DATA_DIR / "premier_league_raw.csv"
    )

    combined_data.to_csv(
        output_path,
        index=False
    )

    print("\n========== DATA SUMMARY ==========")

    print(
        f"Total matches: {len(combined_data)}"
    )

    print(
        f"Total columns: {len(combined_data.columns)}"
    )

    print(
        f"Seasons collected: "
        f"{combined_data['Season'].nunique()}"
    )

    print("\nFirst five records:")
    print(combined_data.head())

    print("\nMissing values:")
    print(combined_data.isnull().sum())

    if failed_seasons:
        print("\nFailed seasons:", failed_seasons)

    print(f"\nDataset saved to: {output_path}")

    return combined_data


if __name__ == "__main__":
    collect_data()

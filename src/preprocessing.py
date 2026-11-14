
import pandas as pd

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "Season",
]

NUMERIC_COLUMNS = [
    "FTHG",
    "FTAG",
    "HTHG",
    "HTAG",
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


def load_data():
    """
    Load historical Premier League match results.
    """

    file_path = (
        RAW_DATA_DIR / "premier_league_raw.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Run python -m src.data_collection first."
        )

    dataframe = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print(f"Initial rows: {len(dataframe)}")
    print(f"Initial columns: {len(dataframe.columns)}")

    return dataframe


def preprocess_data(dataframe):
    """
    Clean and validate historical football data.

    Actual goals and results are retained as targets,
    but must not be used as prediction features
    for the same match.
    """

    df = dataframe.copy()

    print("\n========== PREPROCESSING ==========")

    # 1. Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 2. Standardize text values
    for column in ["HomeTeam", "AwayTeam", "Season"]:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    df["FTR"] = (
        df["FTR"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # 3. Convert match dates
    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce",
    )

    # 4. Convert numeric columns
    available_numeric_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    for column in available_numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # 5. Remove invalid essential information
    initial_rows = len(df)

    df = df.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    print(
        "Rows removed for missing essential values:",
        initial_rows - len(df)
    )

    # 6. Remove invalid team entries
    df = df[
        (df["HomeTeam"] != "") &
        (df["AwayTeam"] != "") &
        (df["HomeTeam"] != df["AwayTeam"])
    ].copy()

    # 7. Remove invalid goal counts
    df = df[
        (df["FTHG"] >= 0) &
        (df["FTAG"] >= 0) &
        (df["FTHG"] % 1 == 0) &
        (df["FTAG"] % 1 == 0)
    ].copy()

    # 8. Verify final result against score
    expected_result = pd.Series(
        "D",
        index=df.index,
        dtype="string"
    )

    expected_result.loc[
        df["FTHG"] > df["FTAG"]
    ] = "H"

    expected_result.loc[
        df["FTHG"] < df["FTAG"]
    ] = "A"

    inconsistent_results = (
        df["FTR"] != expected_result
    )

    print(
        "Rows with inconsistent results:",
        int(inconsistent_results.sum())
    )

    # Remove inconsistent records
    df = df.loc[
        ~inconsistent_results
    ].copy()

    # 9. Remove duplicate fixtures
    duplicate_columns = [
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]

    duplicates = df.duplicated(
        subset=duplicate_columns,
        keep="first"
    )

    print(
        "Duplicate fixtures removed:",
        int(duplicates.sum())
    )

    df = df.loc[~duplicates].copy()

    # 10. Sort matches chronologically
    df = df.sort_values(
        ["Date", "Season", "HomeTeam", "AwayTeam"],
        kind="stable"
    ).reset_index(drop=True)

    # 11. Create classification label
    result_mapping = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win",
    }

    df["ResultLabel"] = df["FTR"].map(
        result_mapping
    )

    # 12. Create regression target
    df["TotalGoals"] = (
        df["FTHG"] + df["FTAG"]
    ).astype(int)

    # 13. Create goal-threshold target
    df["Over2_5"] = (
        df["TotalGoals"] > 2.5
    ).astype(int)

    # 14. Create actual goal difference
    df["ActualGoalDifference"] = (
        df["FTHG"] - df["FTAG"]
    ).astype(int)

    # 15. Add chronological season match order
    df["SeasonMatchNumber"] = (
        df.groupby("Season").cumcount() + 1
    )

    print("\nPreprocessing completed!")
    print("Final rows:", len(df))
    print("Final columns:", len(df.columns))

    return df


def save_processed_data(dataframe):
    """
    Save cleaned data for feature engineering.
    """

    output_file = (
        PROCESSED_DATA_DIR /
        "premier_league_cleaned.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False
    )

    print(f"\nCleaned dataset saved to:\n{output_file}")

    return output_file


def display_summary(dataframe):
    """
    Display basic dataset information.
    """

    print("\n========== DATASET SUMMARY ==========")

    print("\nMatches per season:")
    print(
        dataframe.groupby("Season")
        .size()
        .to_string()
    )

    print("\nMatch result distribution:")
    print(
        dataframe["ResultLabel"]
        .value_counts()
    )

    print("\nAverage goals per match:")
    print(
        round(
            dataframe["TotalGoals"].mean(),
            2
        )
    )

    print("\nOver/Under 2.5 distribution:")
    print(
        dataframe["Over2_5"]
        .value_counts()
    )

    print("\nMissing values:")
    print(
        dataframe.isnull().sum()
    )

    print("\nFirst five cleaned records:")
    print(
        dataframe[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "FTR",
                "TotalGoals",
                "Over2_5",
            ]
        ].head()
    )


def main():
    raw_data = load_data()

    cleaned_data = preprocess_data(raw_data)

    save_processed_data(cleaned_data)

    display_summary(cleaned_data)


if __name__ == "__main__":
    main()

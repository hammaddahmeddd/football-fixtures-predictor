
from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

# Dataset directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Model and output directories
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
for folder in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Historical Premier League dataset source
FOOTBALL_DATA_URL = (
    "https://www.football-data.co.uk/mmz4281"
)

# Five completed Premier League seasons
SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

# English Premier League code
LEAGUE_CODE = "E0"

# API token for upcoming fixtures
FOOTBALL_API_KEY = os.getenv(
    "FOOTBALL_API_KEY", ""
)
CURRENT_SEASON = "2026-27"
CURRENT_SEASON_START_YEAR = 2026

FOOTBALL_API_BASE_URL = (
    "https://api.football-data.org/v4"
)

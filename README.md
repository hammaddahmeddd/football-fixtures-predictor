# Football Fixtures Result Predictor

Comparative machine learning project for Premier League match results, goals, team segmentation and season projections. V4 retains the original Streamlit interface (`app.py`) and uses a separate versioned prediction engine. The experiments are research benchmarks, not guarantees of future results.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install xgboost lightgbm catboost psutil
```

Use Python 3.13.5, the project environment used in development. Copy `.env.example` to `.env` if needed and set `FOOTBALL_API_KEY` locally. Never commit the token. The V4 SVM implementation requires scikit-learn 1.6 or later (development used 1.9.1). Load only trusted generated joblib files.

```powershell
python -m src.production_v4 backtest
python -m src.refresh_v4
python -m streamlit run app.py
```

If an active compatible V4 registry already exists, skip retraining. The refresh command needs API access and the complete current-season schedule. The dashboard reads saved files; refreshing its sidebar does not query the API or train models. Run `python -m pytest tests/ -v` for the existing test suite. Latest V4-specific end-to-end test suite and hosted deployment still require verification.

## Documentation

- [System architecture](docs/architecture.md)
- [Datasets and preprocessing](docs/dataset.md)
- [Algorithms](docs/algorithms.md)
- [Evaluation and results](docs/evaluation.md)
- [Computational complexity](docs/complexity.md)
- [Deployment and reproduction](docs/deployment.md)
- [Limitations and requirement mapping](docs/limitations.md)

## Repository layout

`src/` contains training, experiments and prediction pipelines; `dashboard/` holds original modular pages; `models/v4/` is the versioned registry; `data/` contains raw and processed data; `outputs/` contains experimental metrics, figures and V4 forecasts; `.github/workflows/` contains refresh workflows. V2 remains the deployed fallback until V4 hosting and automation are verified.

# Dataset and preprocessing

| Source | Coverage | Role |
|---|---|---|
| football-data.co.uk E0 | 2011/12–2025/26, 5,700 matches | Historical EPL scores and match statistics |
| football-data.co.uk E1 | 2011/12–2024/25, 7,728 matches | Championship promotion research |
| xgtables.io historical EPL fixtures | 2017/18–2025/26, 3,420 matched fixtures | xG research |
| football-data.org v4 PL competition | 2026/27, changing | Current results and upcoming schedule |

Core fields: Date, Season, HomeTeam, AwayTeam, FTHG, FTAG, FTR (H/D/A). Research optional fields include half-time scores, shots, shots on target, corners and disciplinary counts. Total goals = FTHG + FTAG; Over 2.5 = total >=3. API identifiers and kickoff timestamps identify live forecasts; team names are normalised to historical names. Historical features are constructed before a fixture using prior results, with date-grouped updates, avoiding the current fixture's result in its inputs. First season serves as feature warm-up. Derived features cover rolling form, goal rates, rest, Elo and venue-related performance. Score85, Shot129, Strength159, Promotion169 and XG184 describe research feature sets. Full V4 feature names are stored in the active manifest, not inferred from the feature count. V3.2 feature_dictionary.csv documents its research features.

Scaling is fitted on training data for scale-sensitive models, including KNN, SVM, Poisson and K-Means. Tree models do not need feature standardisation. PCA is used for cluster visualisation rather than assumed as a universal classifier input. Missing research statistics are not silently substituted into V4. Chronological folds train on earlier seasons and evaluate later seasons. The original V2 historical dataset had 1,900 matches; the expanded reference dataset has 5,700. V4 uses 5,320 non-warm-up matches for final supervised fitting if all 15 seasons are present. Record actual training_rows from the manifest. The dataset consists of repeated teams and dependent fixtures, not independent observations.

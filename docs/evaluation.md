# Evaluation and results

Chronological research folds use 2022/23–2023/24 for validation in early experiments; V3.5 and V4 use 2021/22–2022/23 meta-training and 2023/24 selection. The 760 fixtures of 2024/25–2025/26 form a repeatedly examined historical benchmark, not an untouched final test. Report accuracy, macro precision/recall/F1, draw recall, log loss and Brier; regression MAE, RMSE and R²; clustering silhouette, Davies-Bouldin and ARI; RL cumulative reward and convergence. A negative R² means worse squared error than the evaluation-target mean.

| Saved outcome candidate (V3.9) | Accuracy | Macro F1 | Draw recall | Log loss |
|---|---:|---:|---:|---:|
| Always Home | .417105 | .196224 | 0 | NA |
| KNN | .481579 | .364326 | .015228 | 1.032780 |
| SVM | .505263 | .373897 | 0 | 1.026199 |
| Random Forest | .509211 | .381665 | 0 | 1.012939 |
| V3.1 selected | .494737 | .467549 | .314721 | NA |
| V3.3 selected | .473684 | .454747 | .345178 | 1.008496 |
| QAccuracy | .513158 | .388704 | .005076 | 1.002457 |
| V4 primary (separate run) | .500000 | .376926 | not reported | not reported |

V3.6 Decision Tree accuracy .496053, macro F1 .371572. Best observed candidate Shot129_Pois20 .518421 not validation-selected. QAccuracy 390/760 vs validation-selected fixed model 384/760; regret -6. QBalanced 385/760, macro F1 .411341. Compact states selected; 100 and 1,000 passes respectively. See v36_experiments/training_history.csv and convergence graphs for actual convergence rather than inferring it from pass count.

| Goals configuration | MAE | RMSE | R² |
|---|---:|---:|---:|
| SeasonMean | 1.260361 | 1.595754 | -.001156 |
| SeasonMedian / V3.5 MAE | 1.236842 | 1.602629 | -.009802 |
| V3.5 RMSE selected | 1.266899 | 1.598418 | -.004502 |
| V3.3 MAE selected | 1.252632 | 1.635945 | -.052222 |

Best observed Poisson median MAE 1.213158; some shrinkage candidates achieved R² near .0056, but selected advanced estimators did not reliably beat baselines. Over/Under V3.4 selected accuracy .569737, F1 .459682, log loss .683099; prior .557895, F1 .358108, log loss .687474. V3.3 selected F1 .513146. Best observed IP365 accuracy .588158, not selected. V4 selected historical prior and median for both goal objectives, so constant predictions are expected. V4 primary and balanced both ScorePoisson5 boost 1.4; probability model Logistic; exact score DC365.

K-Means V3.7 selected k=2 in all four folds. Training silhouette .3028–.3258, evaluation silhouette .2324–.3457, bootstrap ARI .9717–.9796. Historical RF accuracy .506579 without clusters vs .505263 with; logistic .489474 vs .494737. No consistent predictive improvement established.

V3.9 aligns 760 fixtures, 25 outcome, 9 Over/Under, 9 goal configurations. Uses 2,000 block-bootstrap draws, 28-day blocks within season, McNemar and Holm correction. RF vs AlwaysHome gain .092105, unadjusted block CI [.054167,.129299], adjusted McNemar p=.000033. Adjusted bootstrap p=.106947 for displayed leading tests, not significant at .05. McNemar independence assumption is imperfect. No claim of significant Q advantage without its exact paired-test result. Historical reuse and limited bootstrap resolution constrain inference.

Sources inside repository: `outputs/v39_experiments/full/metrics.csv`, `paired_tests.csv`, `confidence_intervals.csv`, `calibration.csv`; `outputs/v37_experiments/`, `outputs/v36_experiments/`, `outputs/v4_production/<version>/historical_predictions.csv`. Values above transcribed from user-provided terminal output; consult generated CSVs for full metrics and charts.

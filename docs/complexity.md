# Computational complexity

V3.8 benchmark: 17 representative model configurations, 760/2,280/4,940 training fixtures, two repeats, single thread, common 2025/26 evaluation, fresh processes. K-Means uses twice as many profiles as fixtures. Median full workload measurements below; memory is sampled total process RSS, not model-only memory. Feature generation and API costs excluded. These are representative configurations, not exact deployed artifacts.

| Model | Fit s | Single ms | Batch ms | Peak RSS MiB | Model KiB |
|---|---:|---:|---:|---:|---:|
| KNN | .023801 | 1.697425 | 31.61865 | 196.63 | 3322.31 |
| SVM | 1.851629 | 2.806875 | 240.86360 | 227.87 | 2444.36 |
| DecisionTree | .097204 | .138425 | .21485 | 193.48 | 3.92 |
| RandomForest | 2.853535 | 18.784050 | 26.18850 | 196.43 | 2702.34 |
| ExtraTrees | 1.077411 | 19.730900 | 27.61575 | 196.90 | 3448.11 |
| Logistic | .167967 | .411250 | .59345 | 198.79 | 5.31 |
| HistGB | 1.850503 | 3.767350 | 9.26940 | 198.05 | 625.59 |
| XGBoost | 2.630447 | .915250 | 4.30910 | 204.42 | 861.69 |
| LightGBM | 1.939008 | .677575 | 19.04680 | 200.61 | 934.86 |
| CatBoost | 7.953584 | 1.847350 | 2.74965 | 218.17 | 186.13 |
| LinearRegression | .037997 | .302050 | .42050 | 195.43 | 4.34 |
| Ridge | .030028 | .320300 | .44700 | 193.40 | 3.62 |
| Poisson | .029619 | .379475 | .52475 | 195.21 | 3.97 |
| KMeans | .106450 | .451775 | .71475 | 192.15 | 40.21 |
| QLearning | 5.713560 | 24.430275 | 191.53355 | 214.31 | 6654.63 |
| SoftVoting | 5.109139 | 23.664075 | 230.74325 | 230.73 | 8490.97 |
| DixonColes | .072716 | 1.056100 | 21.37365 | 194.96 | 1.56 |

Approximate complexity: n rows, d features, T trees, h depth, I iterations, k clusters, s support vectors, E episodes, A actions, S states. Brute force KNN fit/storage O(nd), query O(nd); kernel SVM fit often O(n²d) to O(n³d), query O(sd), support-vector storage O(sd) (training kernel memory may be quadratic). Balanced tree fit approximately O(d n log n), query O(h), storage O(nodes); random forest multiplies tree complexity by T. Dense least squares O(nd²+d³) for n>=d, query O(d), parameters O(d). Iterative Poisson approximately O(Ind) for first-order evaluations; solver-specific overhead. K-Means O(Inkd), query O(kd), centres O(kd), training data O(nd). Tabular Q update O(1) after action selection, greedy O(A), total O(EA) plus base forecasts, Q table O(SA). Soft voting sums component costs. Dixon-Coles cost depends on optimiser iterations, matches and team parameter count; score-grid inference scales with grid size. Full implementation-specific table is `outputs/v38_experiments/full/theoretical_complexity.csv`.

Decision Tree fastest inference; KNN fastest fit; Dixon-Coles smallest artifact. Soft voting and Q incur multi-model overhead. Compare predictive and runtime metrics only when configuration and evaluation set match. In robotics-like resource-constrained settings, the same accuracy/latency/storage tradeoffs matter, but football benchmarks cannot establish robotics performance.

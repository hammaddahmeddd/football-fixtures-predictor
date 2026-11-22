# Limitations and course requirements

The course asks for five group members, well-documented reproducible source, dataset, PDF report and 15-minute presentation. The project uses the permitted non-robotics context. Confirm actual member names and IDs before the formal report; do not fabricate them.

| Requirement | Evidence |
|---|---|
| Dataset description and cleaning | dataset.md, data_collection, preprocessing, feature engineering |
| Three classifiers | KNN, SVM, Decision Tree; V3/V4 |
| Regression | Linear, Ridge, Poisson; V3.8 and V4 |
| Clustering | V3.7 K-Means, silhouette/DB/ARI and PCA |
| RL | V3.6 Q-Learning, cumulative reward and convergence outputs |
| Ensemble | Random Forest, boosting and soft voting; V3.1/V3.5 |
| Evaluation | V3.9 paired metrics and V3.7/V3.6 specialist outputs |
| Computational analysis | V3.8 measurements, complexity.md |
| Comparative application suitability | evaluation.md and complexity.md, transfer caveats |
| Formal report and presentation | Phases 14 and 15 pending |

Limitations: football outcomes are noisy; models frequently predict no draws; better draw recall may lower accuracy. Advanced regressors have not consistently beaten baselines. V4 historical median and Over prior can produce constant estimates. Modal scorelines and aggregate outcomes differ. xG/shot features are excluded from initial live V4 because consistent inputs are unavailable. Repeatedly inspected evaluation seasons are exploratory. Bootstrap intervals and significance do not establish independent future performance. Archived forecasts are the intended prospective evaluation; no live accuracy claim until settled sample is sufficient. Frozen ratings, independent future matches and random exact ties simplify simulation; injury, transfers, scheduling and in-progress scores are not modelled. Dashboard integration reported complete, automated deployment and actual live settlement remain to be confirmed. Project report and slides should use exact CSV/figure evidence and distinguish observed, validation-selected and live performance.

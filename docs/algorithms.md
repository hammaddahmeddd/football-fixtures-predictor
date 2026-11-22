# Algorithms and implementation

| Algorithm | Method and application |
|---|---|
| KNN | Neighbour voting in scaled feature space; required classifier |
| SVM | Maximum-margin classifier with kernel; V4 uses chronological sigmoid calibration of a frozen fitted estimator |
| Decision Tree | Recursive impurity-based splits; interpretable classifier |
| Random Forest | Bagging and random feature subsets; reduces variance |
| Boosting | XGBoost, LightGBM, CatBoost and histogram gradient boosting explored in research |
| Logistic regression | Multiclass probability estimation |
| Poisson regression | Log-link goal rate estimation separately for home and away; ScorePoisson5 uses alpha=5 |
| Ridge regression | Squared-error regression with L2 penalty; V4 alpha=50 |
| Historical mean/median | Conservative goal baselines for squared/absolute error |
| Dixon-Coles | Time-decayed attack/defence ratings with low-score dependence adjustment; V4 365-day half-life |
| Soft voting | Mean of base probabilities; V4 combines RF, score Poisson, DC365 and IP730 |
| K-Means | Minimises within-cluster squared distances on scaled team profiles; V4 k=2 |
| Q-Learning | One-step contextual model selection, not control of football players |

Poisson probability P(X=k)=exp(-lambda) lambda^k/k!. Sum the score grid across H/D/A or total goals >=3 for outcome/Over probabilities. Dixon-Coles adjusts low-score cells and normalises the grid. Outcome decisions may multiply draw decision scores by a validation-selected factor without changing reported probabilities. K-Means uses 12 pre-match team-profile features, and PCA plots show a two-dimensional projection. Cluster stability is evaluated with bootstrap adjusted Rand index.

Q update: Q(s,a) <- Q(s,a)+alpha[r-Q(s,a)] for gamma=0. The action chooses a model, reward is correctness or a hybrid probability reward in research. The V4 policy uses compact state, accuracy reward, alpha=.15, initial epsilon=.35, 100 passes and seed 42, trained on 2021/22–2022/23 out-of-season predictions. It is not the full 17-action research policy. Monte Carlo samples future scorelines from frozen Dixon-Coles distributions, adds completed standings and ranks teams by points, goal difference and goals scored; exact residual ties are random.


"""
K-Means clustering of Premier League team profiles.

Builds one statistical profile per team per season
using pre-match features from historical fixtures.

Uses only training seasons to select clusters.
"""

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)


TRAIN_SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
]

CLUSTER_FEATURES = [
    "AvgPointsLast5",
    "AvgGoalsScored",
    "AvgGoalsConceded",
    "AvgWinRate",
    "AvgGoalDifference",
]


def load_clustering_data():
    """Load feature-engineered match data."""

    file_path = (
        PROCESSED_DATA_DIR /
        "premier_league_features.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "Feature dataset missing. Run "
            "python -m src.feature_engineering first."
        )

    df = pd.read_csv(file_path)

    df = df[
        df["Season"].isin(TRAIN_SEASONS)
    ].copy()

    required = [
        "Season",
        "HomeTeam",
        "AwayTeam",
        "HomeLast5Points",
        "AwayLast5Points",
        "HomeAvgGoalsScored",
        "AwayAvgGoalsScored",
        "HomeAvgGoalsConceded",
        "AwayAvgGoalsConceded",
        "HomeWinRate",
        "AwayWinRate",
        "HomeGoalDifference",
        "AwayGoalDifference",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing clustering columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            "No training matches available."
        )

    return df


def create_team_profiles(df):
    """
    Convert match-level records into
    team-season statistical profiles.

    Uses pre-match team statistics,
    not the outcomes of the same fixture.
    """

    home = pd.DataFrame({
        "Season": df["Season"],
        "Team": df["HomeTeam"],
        "AvgPointsLast5": df["HomeLast5Points"],
        "AvgGoalsScored": df["HomeAvgGoalsScored"],
        "AvgGoalsConceded": df["HomeAvgGoalsConceded"],
        "AvgWinRate": df["HomeWinRate"],
        "AvgGoalDifference": df["HomeGoalDifference"],
    })

    away = pd.DataFrame({
        "Season": df["Season"],
        "Team": df["AwayTeam"],
        "AvgPointsLast5": df["AwayLast5Points"],
        "AvgGoalsScored": df["AwayAvgGoalsScored"],
        "AvgGoalsConceded": df["AwayAvgGoalsConceded"],
        "AvgWinRate": df["AwayWinRate"],
        "AvgGoalDifference": df["AwayGoalDifference"],
    })

    combined = pd.concat(
        [home, away],
        ignore_index=True
    )

    profiles = (
        combined.groupby(
            ["Season", "Team"],
            as_index=False
        )[CLUSTER_FEATURES]
        .mean()
    )

    if profiles[CLUSTER_FEATURES].isnull().any().any():
        raise ValueError(
            "Missing team profile values."
        )

    print(
        f"Created {len(profiles)} "
        "team-season profiles."
    )

    return profiles


def choose_best_k(X_scaled):
    """
    Compare cluster counts using silhouette score.
    """

    scores = []

    max_k = min(6, len(X_scaled) - 1)

    if max_k < 2:
        raise ValueError(
            "Not enough team profiles for clustering."
        )

    print("\n===== CLUSTER EVALUATION =====")

    for k in range(2, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        scores.append({
            "Clusters": k,
            "SilhouetteScore": float(score),
        })

        print(
            f"K={k} | Silhouette={score:.4f}"
        )

    results_df = pd.DataFrame(scores)

    best_row = results_df.loc[
        results_df["SilhouetteScore"].idxmax()
    ]

    best_k = int(best_row["Clusters"])

    print(f"\nBest K: {best_k}")

    return best_k, results_df


def save_cluster_graphs(
    X_scaled,
    labels,
    score_df,
):
    """Save PCA and silhouette score plots."""

    graph_dir = OUTPUTS_DIR / "graphs"
    graph_dir.mkdir(
        parents=True, exist_ok=True
    )

    # PCA is only used for 2D visualization.
    pca = PCA(
        n_components=2,
        random_state=42
    )

    reduced_data = pca.fit_transform(
        X_scaled
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.8,
        s=65
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Premier League Team Clusters")

    fig.colorbar(
        scatter,
        ax=ax,
        label="Cluster"
    )

    fig.tight_layout()

    fig.savefig(
        graph_dir / "team_clusters_pca.png",
        dpi=150
    )

    plt.close(fig)

    # Silhouette score comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        score_df["Clusters"],
        score_df["SilhouetteScore"],
        marker="o"
    )

    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("K-Means Cluster Evaluation")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        graph_dir / "kmeans_silhouette_scores.png",
        dpi=150
    )

    plt.close(fig)

    print("\nClustering graphs saved.")


def train_kmeans():
    """Train and evaluate K-Means clustering."""

    df = load_clustering_data()

    profiles = create_team_profiles(df)

    X = profiles[CLUSTER_FEATURES]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    best_k, score_df = choose_best_k(
        X_scaled
    )

    # Fit final K-Means pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "kmeans",
            KMeans(
                n_clusters=best_k,
                random_state=42,
                n_init=10
            )
        )
    ])

    model.fit(X)

    labels = model.predict(X)

    profiles["Cluster"] = labels

    final_score = silhouette_score(
        X_scaled,
        labels
    )

    print("\n===== FINAL CLUSTERING RESULTS =====")
    print(f"Selected clusters: {best_k}")
    print(f"Silhouette score: {final_score:.4f}")

    print("\nCluster sizes:")
    print(
        profiles["Cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_summary = (
        profiles.groupby("Cluster")[
            CLUSTER_FEATURES
        ].mean()
    )

    print("\nCluster characteristics:")
    print(cluster_summary.round(3))

    # Save model
    MODELS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(
        model,
        MODELS_DIR / "kmeans_model.joblib"
    )

    joblib.dump(
        CLUSTER_FEATURES,
        MODELS_DIR / "cluster_features.joblib"
    )

    # Save outputs
    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(
        parents=True, exist_ok=True
    )

    profiles.to_csv(
        metrics_dir / "team_clusters.csv",
        index=False
    )

    score_df.to_csv(
        metrics_dir / "clustering_results.csv",
        index=False
    )

    cluster_summary.to_csv(
        metrics_dir / "cluster_summary.csv"
    )

    # Save visualizations
    save_cluster_graphs(
        X_scaled,
        labels,
        score_df
    )

    print("\n===== COMPLETED =====")
    print("K-Means model saved.")
    print("Cluster results saved.")
    print("Clustering graphs saved.")

    return model


if __name__ == "__main__":
    train_kmeans()

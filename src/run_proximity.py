#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import theilslopes
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from common import DATA, FIGURES, TABLES, ensure_output_dirs, read_table, write_manifest
from run_network import harmonize, likely_corporate


def annual_portfolios(raw: pd.DataFrame, minimum: float, firms: set[str] | None = None) -> dict[int, pd.DataFrame]:
    raw = raw.copy()
    raw["assignee"] = harmonize(raw["assignee"])
    totals = raw.groupby(["publication_year", "assignee"])["fractional_family_count"].sum()
    keep = totals[totals >= minimum].index
    if firms is not None:
        # Only the leading firms enter the similarity calculation; restricting the pivot to
        # them leaves every cosine similarity unchanged and keeps memory use small.
        keep = keep[keep.get_level_values("assignee").isin(sorted(firms))]
    raw = raw.set_index(["publication_year", "assignee"]).loc[keep].reset_index()
    out = {}
    for year, part in raw.groupby("publication_year"):
        matrix = part.pivot_table(index="assignee", columns="cpc_code", values="fractional_family_count", aggfunc="sum", fill_value=0.0)
        out[int(year)] = matrix
    return out


def pairwise_similarities(portfolios: dict[int, pd.DataFrame], top_firms: set[str]) -> pd.DataFrame:
    records = []
    for year, matrix in portfolios.items():
        matrix = matrix.loc[matrix.index.intersection(sorted(top_firms))]
        x = normalize(matrix.to_numpy(dtype=float), norm="l2")
        sim = x @ x.T
        for i, j in combinations(range(len(matrix)), 2):
            records.append({"year": year, "firm_a": matrix.index[i], "firm_b": matrix.index[j], "similarity": float(sim[i, j])})
    return pd.DataFrame(records)


def pair_fixed_effect_trend(pair_year: pd.DataFrame, permutations: int, seed: int) -> dict:
    if pair_year.empty:
        raise ValueError("No eligible firm pairs remain after filtering.")
    pair_year = pair_year.copy()
    pair_year["pair"] = pair_year["firm_a"] + " | " + pair_year["firm_b"]
    pair_year["year_centered"] = pair_year["year"] - pair_year.groupby("pair")["year"].transform("mean")
    pair_year["similarity_within_pair"] = pair_year["similarity"] - pair_year.groupby("pair")["similarity"].transform("mean")
    denominator = np.square(pair_year["year_centered"]).sum()
    beta = float((pair_year["year_centered"] * pair_year["similarity_within_pair"]).sum() / denominator)

    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    groups = [group.index.to_numpy() for _, group in pair_year.groupby("pair")]
    for i in range(permutations):
        shuffled_similarity = pair_year["similarity_within_pair"].to_numpy().copy()
        for indexes in groups:
            shuffled_similarity[indexes] = rng.permutation(shuffled_similarity[indexes])
        null[i] = (pair_year["year_centered"].to_numpy() * shuffled_similarity).sum() / denominator
    return {
        "annual_change_in_cosine_similarity": beta,
        "permutation_p_two_sided": float((1 + (np.abs(null) >= abs(beta)).sum()) / (permutations + 1)),
        "null_2_5_percentile": float(np.quantile(null, 0.025)),
        "null_97_5_percentile": float(np.quantile(null, 0.975)),
        "observations": int(len(pair_year)),
        "unique_pairs": int(pair_year["pair"].nunique()),
    }


def stable_clustering(pooled: pd.DataFrame, k_values: list[int], bootstrap: int, seed: int) -> tuple:
    matrix = pooled.pivot_table(index="assignee", columns="cpc_code", values="fractional_family_count", aggfunc="sum", fill_value=0.0)
    x = normalize(matrix.to_numpy(dtype=float), norm="l2")
    distances = pdist(x, metric="euclidean")
    tree = linkage(distances, method="ward")
    candidates = []
    labels_by_k = {}
    for k in k_values:
        labels = fcluster(tree, k, criterion="maxclust")
        labels_by_k[k] = labels
        candidates.append({"k": k, "silhouette": silhouette_score(x, labels, metric="euclidean")})
    candidate_df = pd.DataFrame(candidates)
    best_k = int(candidate_df.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])

    rng = np.random.default_rng(seed)
    ari = []
    base = labels_by_k[best_k]
    for _ in range(bootstrap):
        # Uniform feature bootstrap: weighting selection probabilities by CPC
        # frequency would count prevalent codes twice (in selection and values).
        sampled_features = rng.choice(matrix.columns, size=len(matrix.columns), replace=True)
        xb = normalize(matrix.loc[:, sampled_features].to_numpy(dtype=float), norm="l2")
        if np.unique(xb, axis=0).shape[0] < best_k:
            continue
        boot_tree = linkage(pdist(xb, metric="euclidean"), method="ward")
        boot_labels = fcluster(boot_tree, best_k, criterion="maxclust")
        ari.append(adjusted_rand_score(base, boot_labels))
    assignments = pd.DataFrame({"assignee": matrix.index, "cluster": base})
    stability = {
        "selected_k": best_k,
        "bootstrap_repetitions_used": len(ari),
        "median_adjusted_rand_index": float(np.median(ari)),
        "ari_2_5_percentile": float(np.quantile(ari, 0.025)),
        "ari_97_5_percentile": float(np.quantile(ari, 0.975)),
    }
    return assignments, candidate_df, stability, tree, matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA / "assignee_cpc_year.parquet")
    parser.add_argument("--minimum-families", type=float, default=20.0)
    parser.add_argument("--top-firms", type=int, default=30)
    parser.add_argument("--organization-scope", choices=["all", "corporate"], default="corporate")
    parser.add_argument("--label", default="corporate")
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--g06n-only", action="store_true", help="Use only G06N hierarchy features, excluding co-classifications.")
    parser.add_argument("--cpc-level", choices=["full", "subclass"], default="full")
    parser.add_argument(
        "--weight-mode",
        choices=["fractional_family", "code_occurrence"],
        default="fractional_family",
        help="Allocate unit family mass across CPCs, or count every CPC occurrence while fractionalizing only multiple assignees.",
    )
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=266665684)
    args = parser.parse_args()
    ensure_output_dirs()

    raw = read_table(args.input)
    required = {"family_id", "publication_year", "assignee", "cpc_code", "fractional_family_count"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    raw = raw[raw["publication_year"] <= args.end_year].copy()
    if args.g06n_only:
        if "is_g06n" not in raw.columns:
            raise ValueError("--g06n-only requires the is_g06n column.")
        raw = raw[raw["is_g06n"].astype(bool)].copy()
    if args.cpc_level == "subclass":
        raw["cpc_code"] = raw["cpc_code"].astype(str).str[:4]
        raw = (
            raw.groupby(["family_id", "publication_year", "assignee", "cpc_code"], as_index=False)["fractional_family_count"]
            .max()
        )
    if raw.empty:
        raise ValueError("No rows remain after the requested year/CPC filters.")
    raw["assignee"] = harmonize(raw["assignee"])
    # Recompute weights after alias consolidation so spelling variants within a
    # family cannot receive duplicated mass.
    raw = raw.drop_duplicates(["family_id", "publication_year", "assignee", "cpc_code"]).copy()
    assignee_count = raw.groupby("family_id")["assignee"].transform("nunique")
    cpc_count = raw.groupby("family_id")["cpc_code"].transform("nunique")
    if args.weight_mode == "fractional_family":
        raw["fractional_family_count"] = 1.0 / (assignee_count * cpc_count)
    else:
        raw["fractional_family_count"] = 1.0 / assignee_count
    family_counts = raw[["family_id", "assignee"]].drop_duplicates().groupby("assignee")["family_id"].nunique().sort_values(ascending=False)
    if args.organization_scope == "corporate":
        family_counts = family_counts[likely_corporate(pd.Series(family_counts.index, index=family_counts.index)).to_numpy()]
    firm_totals = family_counts.nlargest(args.top_firms)
    firms = set(firm_totals.index)

    portfolios = annual_portfolios(raw, args.minimum_families, firms)
    similarities = pairwise_similarities(portfolios, firms)
    similarity_path = TABLES / f"annual_pairwise_similarity_{args.label}.csv"
    similarities.to_csv(similarity_path, index=False)
    trend = pair_fixed_effect_trend(similarities, args.permutations, args.seed)
    trend["organization_scope"] = args.organization_scope
    trend["top_assignees"] = args.top_firms
    trend["end_year"] = args.end_year
    trend["g06n_only"] = args.g06n_only
    trend["cpc_level"] = args.cpc_level
    trend["weight_mode"] = args.weight_mode
    trend_path = TABLES / f"h2_convergence_test_{args.label}.json"
    trend_path.write_text(json.dumps(trend, indent=2), encoding="utf-8")

    annual_summary = similarities.groupby("year")["similarity"].agg(["mean", "median", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=annual_summary, x="year", y="mean", marker="o", ax=ax)
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("Annual technological proximity among leading G06N assignees")
    fig.tight_layout()
    trend_figure = FIGURES / f"annual_proximity_trend_{args.label}.png"
    fig.savefig(trend_figure, dpi=300)
    plt.close(fig)

    pooled = raw[raw["assignee"].isin(firms)].groupby(["family_id", "assignee", "cpc_code"], as_index=False)["fractional_family_count"].sum()
    assignments, candidates, stability, tree, matrix = stable_clustering(pooled, [2, 3, 4, 5, 6], args.bootstrap, args.seed)
    assignments_path = TABLES / f"cluster_assignments_{args.label}.csv"
    candidates_path = TABLES / f"cluster_selection_{args.label}.csv"
    stability_path = TABLES / f"cluster_stability_{args.label}.json"
    assignments.to_csv(assignments_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    stability_path.write_text(json.dumps(stability, indent=2), encoding="utf-8")

    cluster_profile = raw.merge(assignments, on="assignee", how="inner")
    cluster_profile = cluster_profile.groupby(["cluster", "cpc_code"], as_index=False)["fractional_family_count"].sum()
    cluster_totals = cluster_profile.groupby("cluster")["fractional_family_count"].transform("sum")
    overall = raw[raw["assignee"].isin(firms)].groupby("cpc_code")["fractional_family_count"].sum()
    overall_total = float(overall.sum())
    cluster_profile["within_cluster_share"] = cluster_profile["fractional_family_count"] / cluster_totals
    cluster_profile["overall_share"] = cluster_profile["cpc_code"].map(overall) / overall_total
    cluster_profile["location_quotient"] = cluster_profile["within_cluster_share"] / cluster_profile["overall_share"]
    cluster_profile["within_cluster_rank"] = cluster_profile.groupby("cluster")["fractional_family_count"].rank(method="first", ascending=False)
    cluster_profile = cluster_profile[cluster_profile["within_cluster_rank"] <= 20].sort_values(["cluster", "within_cluster_rank"])
    profile_path = TABLES / f"cluster_cpc_profiles_{args.label}.csv"
    cluster_profile.to_csv(profile_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    dendrogram(tree, labels=matrix.index.tolist(), leaf_rotation=90, ax=ax)
    ax.set_ylabel("Ward linkage distance on L2-normalized CPC vectors")
    ax.set_title("Strategic clustering of leading G06N assignees")
    fig.tight_layout()
    dendrogram_path = FIGURES / f"assignee_dendrogram_{args.label}.png"
    fig.savefig(dendrogram_path, dpi=300)
    plt.close(fig)

    outputs = [similarity_path, trend_path, trend_figure, assignments_path, candidates_path, stability_path, profile_path, dendrogram_path]
    write_manifest(f"proximity_{args.label}", [args.input], vars(args), outputs)


if __name__ == "__main__":
    main()

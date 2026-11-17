#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from common import DATA, FIGURES, TABLES, ensure_output_dirs, gini, hhi, read_table, write_manifest


ALIASES = {
    # Keep this list explicit and evidence-auditable. Expand only after checking names.
    "MICROSOFT TECHNOLOGY LICENSING LLC": "MICROSOFT",
    "MICROSOFT CORP": "MICROSOFT",
    "MICROSOFT CORPORATION": "MICROSOFT",
    "GOOGLE LLC": "GOOGLE",
    "GOOGLE INC": "GOOGLE",
    "INTERNATIONAL BUSINESS MACHINES CORPORATION": "IBM",
    "IBM CORP": "IBM",
    "SAMSUNG ELECTRONICS CO LTD": "SAMSUNG ELECTRONICS",
    "MICROSOFT TECH PERMISSION LIMITED RESPONSIBILITY COMPANY": "MICROSOFT",
    "MICROSOFT LICENSING TECH LLC": "MICROSOFT",
    "MICROSOFT TECH LICESNING LLC": "MICROSOFT",
    "MICROSOFT TECHNOLOGY LICENSING": "MICROSOFT",
    "MICROSOFT TEHCNOLOGY LICENSING LLC": "MICROSOFT",
    "MICROSOFT TECHNOIOGY LICENSING LLC": "MICROSOFT",
    "GOOGLE LTD RESPONSIBILITY COMPANY": "GOOGLE",
    "GOOGLE INC GOOGLE LLC": "GOOGLE",
    "GOOGLE LLC GOOGLE INC": "GOOGLE",
    "GOOGLE LCC": "GOOGLE",
    "FACEBOOK INC": "META PLATFORMS",
    "FACEBOOK TECH LLC": "META PLATFORMS",
    "FACEBOOK TECH INC": "META PLATFORMS",
    "FACEBOOK TECHNOLOGIES LLC": "META PLATFORMS",
    "META PLATFORMS INC": "META PLATFORMS",
    "META PLATFORMS TECH LLC": "META PLATFORMS",
    "AMAZON TECH INC": "AMAZON",
    "AMAZON COM INC": "AMAZON",
    "BEIJING BAIDU NETCOM SCI TEC": "BEIJING BAIDU NETCOM SCI TECH CO LTD",
}


def harmonize(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.upper().str.replace(r"[^A-Z0-9]+", " ", regex=True).str.strip()
    return cleaned.replace(ALIASES)


NON_CORPORATE_PATTERN = (
    r"\b(?:UNIV|UNIVERSITY|COLLEGE|SCHOOL|ACADEMY|INSTITUTE|INST|LAB|LABORATORY|"
    r"HOSPITAL|MINISTRY|GOVERNMENT|FOUNDATION|RESEARCH CENTER|RESEARCH CENTRE)\b"
)


def likely_corporate(series: pd.Series) -> pd.Series:
    """Transparent name-based screen; results must also be reported without it."""
    return ~series.astype(str).str.contains(NON_CORPORATE_PATTERN, case=False, regex=True, na=False)


def graph_from_edges(df: pd.DataFrame) -> nx.DiGraph:
    grouped = df.groupby(["citing_assignee", "cited_assignee"], as_index=False)["fractional_weight"].sum()
    graph = nx.DiGraph()
    for row in grouped.itertuples(index=False):
        graph.add_edge(row.citing_assignee, row.cited_assignee, weight=float(row.fractional_weight))
    return graph


def centrality_table(graph: nx.DiGraph) -> pd.DataFrame:
    in_strength = dict(graph.in_degree(weight="weight"))
    out_strength = dict(graph.out_degree(weight="weight"))
    pagerank = nx.pagerank(graph, weight="weight")
    distance_graph = graph.copy()
    for _, _, data in distance_graph.edges(data=True):
        data["distance"] = 1.0 / max(float(data["weight"]), 1e-12)
    betweenness = nx.betweenness_centrality(distance_graph, weight="distance", normalized=True)
    rows = []
    for node in graph.nodes:
        rows.append({
            "assignee": node,
            "in_strength": in_strength.get(node, 0.0),
            "out_strength": out_strength.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0),
            "betweenness": betweenness.get(node, 0.0),
        })
    return pd.DataFrame(rows).sort_values(["in_strength", "pagerank"], ascending=False)


def max_weighted_betweenness(graph: nx.DiGraph) -> float:
    distance_graph = graph.copy()
    for _, _, data in distance_graph.edges(data=True):
        data["distance"] = 1.0 / max(float(data["weight"]), 1e-12)
    values = nx.betweenness_centrality(distance_graph, weight="distance", normalized=True)
    return float(max(values.values(), default=0.0))


def brokerage_null(graph: nx.DiGraph, repetitions: int, seed: int, maximum_nodes: int = 100) -> dict:
    """Degree-preserving directed rewiring test on the strongest-node induced graph."""
    rng = np.random.default_rng(seed)
    strength = {node: graph.in_degree(node, weight="weight") + graph.out_degree(node, weight="weight") for node in graph}
    nodes = sorted(strength, key=strength.get, reverse=True)[:maximum_nodes]
    observed_graph = graph.subgraph(nodes).copy()
    observed = max_weighted_betweenness(observed_graph)
    base_edges = list(observed_graph.edges())
    base_weights = [observed_graph[u][v]["weight"] for u, v in base_edges]
    null = []
    for _ in range(repetitions):
        randomized = observed_graph.copy()
        edge_list = list(randomized.edges())
        attempts = 0
        accepted = 0
        target = max(1, len(base_edges))
        while attempts < 50 * target and accepted < target:
            attempts += 1
            first, second = rng.choice(len(edge_list), size=2, replace=False)
            a, b = edge_list[first]
            c, d = edge_list[second]
            if len({a, b, c, d}) < 4 or randomized.has_edge(a, d) or randomized.has_edge(c, b):
                continue
            wab = randomized[a][b]["weight"]
            wcd = randomized[c][d]["weight"]
            randomized.remove_edge(a, b)
            randomized.remove_edge(c, d)
            randomized.add_edge(a, d, weight=wab)
            randomized.add_edge(c, b, weight=wcd)
            edge_list[first] = (a, d)
            edge_list[second] = (c, b)
            accepted += 1
        shuffled_weights = rng.permutation(base_weights)
        for (u, v), weight in zip(randomized.edges(), shuffled_weights):
            randomized[u][v]["weight"] = float(weight)
        null.append(max_weighted_betweenness(randomized))
    null_values = np.asarray(null)
    return {
        "induced_nodes": len(nodes),
        "observed_max_betweenness": observed,
        "null_mean": float(null_values.mean()),
        "null_2_5_percentile": float(np.quantile(null_values, 0.025)),
        "null_97_5_percentile": float(np.quantile(null_values, 0.975)),
        "upper_tail_p": float((1 + (null_values >= observed).sum()) / (repetitions + 1)),
        "repetitions": repetitions,
        "interpretation_limit": "The null test is a topology sensitivity analysis on the strongest-node induced network; it does not establish causal control of knowledge flows.",
    }


def bootstrap_ranks(edges: pd.DataFrame, nodes: list[str], repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    family_codes, families = pd.factorize(edges["citing_family_id"], sort=False)
    if len(families) == 0 or repetitions <= 0:
        return pd.DataFrame()
    node_index = {node: i for i, node in enumerate(nodes)}
    source = edges["citing_assignee"].map(node_index).to_numpy()
    target = edges["cited_assignee"].map(node_index).to_numpy()
    edge_code = source * len(nodes) + target
    base_weight = edges["fractional_weight"].to_numpy(dtype=float)
    records = []
    for rep in range(repetitions):
        multiplicity = rng.poisson(1.0, size=len(families))
        weights = np.bincount(
            edge_code,
            weights=base_weight * multiplicity[family_codes],
            minlength=len(nodes) * len(nodes),
        ).reshape((len(nodes), len(nodes)))
        graph = nx.from_numpy_array(weights, create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, dict(enumerate(nodes)))
        in_strength = np.asarray(weights.sum(axis=0)).ravel()
        pagerank = nx.pagerank(graph, weight="weight")
        scores = pd.DataFrame({"assignee": nodes, "in_strength": in_strength, "pagerank": [pagerank[node] for node in nodes]})
        scores["rank_in_strength"] = scores["in_strength"].rank(method="min", ascending=False)
        scores["replicate"] = rep
        records.append(scores[["assignee", "in_strength", "pagerank", "rank_in_strength", "replicate"]])
    boot = pd.concat(records, ignore_index=True)
    return boot.groupby("assignee").agg(
        in_strength_low=("in_strength", lambda x: x.quantile(0.025)),
        in_strength_high=("in_strength", lambda x: x.quantile(0.975)),
        pagerank_low=("pagerank", lambda x: x.quantile(0.025)),
        pagerank_high=("pagerank", lambda x: x.quantile(0.975)),
        median_rank=("rank_in_strength", "median"),
        top10_probability=("rank_in_strength", lambda x: float((x <= 10).mean())),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA / "citation_edges.parquet")
    parser.add_argument("--portfolio-input", type=Path, default=DATA / "assignee_cpc_year.parquet")
    parser.add_argument("--minimum-families", type=int, default=50)
    parser.add_argument("--maximum-nodes", type=int, default=300)
    parser.add_argument("--organization-scope", choices=["all", "corporate"], default="corporate")
    parser.add_argument("--label", default="corporate")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--null", type=int, default=100)
    parser.add_argument("--seed", type=int, default=266665684)
    args = parser.parse_args()
    ensure_output_dirs()

    edges = pd.read_parquet(args.input, columns=["citing_family_id", "citing_assignee", "cited_assignee", "fractional_weight"])
    required = {"citing_family_id", "citing_assignee", "cited_assignee", "fractional_weight"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    edges = edges.dropna(subset=["citing_assignee", "cited_assignee", "fractional_weight"]).copy()
    edges["citing_assignee"] = harmonize(edges["citing_assignee"])
    edges["cited_assignee"] = harmonize(edges["cited_assignee"])
    edges = edges[edges["citing_assignee"] != edges["cited_assignee"]]

    portfolio = pd.read_parquet(args.portfolio_input, columns=["family_id", "assignee"]).drop_duplicates()
    portfolio["assignee"] = harmonize(portfolio["assignee"])
    family_counts = portfolio.groupby("assignee")["family_id"].nunique().sort_values(ascending=False)
    eligible = family_counts[family_counts >= args.minimum_families]
    if args.organization_scope == "corporate":
        eligible = eligible[likely_corporate(pd.Series(eligible.index, index=eligible.index)).to_numpy()]
    eligible_nodes = set(eligible.index)
    eligible_edges = edges[edges["citing_assignee"].isin(eligible_nodes) & edges["cited_assignee"].isin(eligible_nodes)].copy()
    selected_nodes = eligible.head(args.maximum_nodes).index.tolist()
    selected = set(selected_nodes)
    edges = eligible_edges[eligible_edges["citing_assignee"].isin(selected) & eligible_edges["cited_assignee"].isin(selected)].copy()

    graph = graph_from_edges(edges)
    centrality = centrality_table(graph)
    boot = bootstrap_ranks(edges, sorted(graph.nodes()), args.bootstrap, args.seed)
    if not boot.empty:
        centrality = centrality.merge(boot, on="assignee", how="left")
    centrality_path = TABLES / f"network_centrality_{args.label}.csv"
    centrality.to_csv(centrality_path, index=False)

    eligible_strengths = eligible_edges.groupby("cited_assignee")["fractional_weight"].sum().reindex(eligible.index, fill_value=0.0).to_numpy()
    strengths = eligible_strengths
    concentration = {
        "eligible_nodes": int(len(eligible)),
        "centrality_graph_nodes": graph.number_of_nodes(),
        "centrality_graph_directed_edges": graph.number_of_edges(),
        "total_fractional_citation_weight": float(strengths.sum()),
        "in_strength_gini": gini(strengths),
        "in_strength_hhi": hhi(strengths),
        "top_1_share": float(np.sort(strengths)[-1:].sum() / strengths.sum()),
        "top_5_share": float(np.sort(strengths)[-5:].sum() / strengths.sum()),
        "top_10_share": float(np.sort(strengths)[-10:].sum() / strengths.sum()),
        "organization_scope": args.organization_scope,
        "minimum_g06n_families": args.minimum_families,
        "corporate_screen_limit": "Name-based classification; the all-assignee sensitivity analysis is required.",
    }
    concentration_path = TABLES / f"network_concentration_{args.label}.json"
    concentration_path.write_text(json.dumps(concentration, indent=2), encoding="utf-8")
    null_result = brokerage_null(graph, args.null, args.seed)
    null_path = TABLES / f"h1_brokerage_null_test_{args.label}.json"
    null_path.write_text(json.dumps(null_result, indent=2), encoding="utf-8")

    top = centrality.nlargest(30, "in_strength")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(top["pagerank"], top["betweenness"], s=30 + 300 * top["in_strength"] / top["in_strength"].max())
    for row in top.itertuples(index=False):
        ax.annotate(row.assignee, (row.pagerank, row.betweenness), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("PageRank (prestige)")
    ax.set_ylabel("Weighted betweenness centrality (brokerage)")
    ax.set_title("Strategic positioning in the G06N inter-assignee citation network")
    fig.tight_layout()
    figure_path = FIGURES / f"network_positioning_{args.label}.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    write_manifest(f"network_{args.label}", [args.input, args.portfolio_input], vars(args) | {"aliases": ALIASES}, [centrality_path, concentration_path, null_path, figure_path])


if __name__ == "__main__":
    main()

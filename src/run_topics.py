#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score
from umap import UMAP
from hdbscan import HDBSCAN

from common import DATA, FIGURES, RESULTS, TABLES, ensure_output_dirs, read_table, write_manifest


GENERATIVE_TERMS = {
    "generative", "generation", "generate", "prompt", "transformer", "language",
    "text", "semantic", "foundation", "large language", "llm", "diffusion",
}
VISION_TERMS = {
    "image", "vision", "object", "recognition", "video", "visual",
    "segmentation", "detection", "pixel", "camera", "face",
}


def build_model(seed: int, model_name: str, minimum_topic_size: int, minimum_samples: int, cluster_selection_method: str) -> BERTopic:
    return BERTopic(
        embedding_model=None,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(
            min_cluster_size=minimum_topic_size,
            min_samples=minimum_samples,
            metric="euclidean",
            cluster_selection_method=cluster_selection_method,
            prediction_data=True,
        ),
        calculate_probabilities=False,
        verbose=True,
    )


def topic_term_set(model: BERTopic, topic: int, n: int = 15) -> set[str]:
    return {term.lower() for term, _ in (model.get_topic(topic) or [])[:n]}


def identify_topic_set(model: BERTopic, target_terms: set[str], label: str) -> tuple[list[int], pd.DataFrame]:
    rows = []
    for topic in model.get_topics():
        if topic == -1:
            continue
        terms = topic_term_set(model, topic, 20)
        overlap = sorted(term for term in target_terms if term in terms)
        rows.append({"target": label, "topic": topic, "matched_terms": "; ".join(overlap), "match_count": len(overlap), "top_terms": "; ".join(sorted(terms))})
    scores = pd.DataFrame(rows).sort_values(["match_count", "topic"], ascending=[False, True])
    if scores.empty:
        return [], scores
    return scores.loc[scores["match_count"] >= 2, "topic"].astype(int).tolist(), scores


def annual_prevalence(frame: pd.DataFrame, topic_column: str = "topic") -> pd.DataFrame:
    counts = frame.groupby(["publication_year", topic_column]).size().rename("documents").reset_index()
    totals = frame.groupby("publication_year").size().rename("year_total").reset_index()
    counts = counts.merge(totals, on="publication_year")
    counts["within_year_share"] = counts["documents"] / counts["year_total"]
    return counts


def post_2023_slope_change_test(series: pd.DataFrame, permutations: int, seed: int) -> dict:
    periods = series["period_index"].to_numpy(dtype=float)
    y = series["within_period_share"].to_numpy(dtype=float)
    break_index = float(series.loc[series["period_label"] == "2023Q1", "period_index"].iloc[0])
    post = np.maximum(periods - break_index, 0.0)
    x = np.column_stack([np.ones(len(periods)), periods - periods.min(), post])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    observed = float(beta[2])
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for i in range(permutations):
        null[i] = np.linalg.lstsq(x, rng.permutation(y), rcond=None)[0][2]
    return {
        "prespecified_break_period": "2023Q1",
        "pre_2023_quarterly_slope": float(beta[1]),
        "post_2023_incremental_quarterly_slope": observed,
        "post_2023_total_quarterly_slope": float(beta[1] + beta[2]),
        "permutation_p_two_sided": float((1 + (np.abs(null) >= abs(observed)).sum()) / (permutations + 1)),
        "periods": series["period_label"].tolist(),
        "shares": y.tolist(),
        "warning": "Quarterly publication prevalence remains observational and can be affected by publication and classification lags.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA / "abstract_sample.parquet")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--minimum-topic-size", type=int, default=100)
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="leaf")
    parser.add_argument("--reduce-outliers-threshold", type=float, default=0.2)
    parser.add_argument("--seeds", default="266665684,266665685,266665686,266665687,266665688")
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--recompute-embeddings", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()

    frame = read_table(args.input)
    required = {"family_id", "publication_year", "publication_date", "abstract_en"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["abstract_en"]).sort_values(["publication_year", "family_id"]).reset_index(drop=True)
    documents = frame["abstract_en"].astype(str).tolist()
    seeds = [int(item) for item in args.seeds.split(",")]
    sample_fingerprint = hashlib.sha256("\n".join(frame["family_id"].astype(str)).encode("utf-8")).hexdigest()

    embedding_path = RESULTS / "abstract_embeddings.npy"
    embedding_metadata_path = RESULTS / "abstract_embeddings_metadata.json"
    if embedding_path.exists() and not args.recompute_embeddings:
        if not embedding_metadata_path.exists():
            raise ValueError("Embedding metadata is missing; rerun with --recompute-embeddings.")
        embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
        if embedding_metadata.get("sample_family_sha256") != sample_fingerprint:
            raise ValueError("Existing embeddings belong to a different family sample; rerun with --recompute-embeddings.")
        embeddings = np.load(embedding_path)
        if embeddings.shape[0] != len(documents):
            raise ValueError("Existing embeddings do not match the active abstract sample; rerun with --recompute-embeddings.")
    else:
        encoder = SentenceTransformer(args.model)
        embeddings = encoder.encode(documents, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        np.save(embedding_path, embeddings)
        embedding_metadata_path.write_text(
            json.dumps({"sample_family_sha256": sample_fingerprint, "rows": len(frame), "model": args.model}, indent=2),
            encoding="utf-8",
        )

    models = []
    assignments = []
    generative_topics = []
    vision_topics = []
    raw_outlier_counts = []
    remaining_outlier_counts = []
    score_tables = []
    for seed in seeds:
        model = build_model(seed, args.model, args.minimum_topic_size, args.minimum_samples, args.cluster_selection_method)
        raw_topics, _ = model.fit_transform(documents, embeddings)
        raw_outlier_counts.append(int((np.asarray(raw_topics) == -1).sum()))
        topics = model.reduce_outliers(
            documents,
            raw_topics,
            strategy="embeddings",
            embeddings=embeddings,
            threshold=args.reduce_outliers_threshold,
        )
        model.update_topics(documents, topics=topics)
        remaining_outlier_counts.append(int((np.asarray(topics) == -1).sum()))
        models.append(model)
        assignments.append(np.asarray(topics))
        generative_set, generative_scores = identify_topic_set(model, GENERATIVE_TERMS, "generative_ai")
        vision_set, vision_scores = identify_topic_set(model, VISION_TERMS, "vision")
        generative_topics.append(generative_set)
        vision_topics.append(vision_set)
        for scores in (generative_scores, vision_scores):
            scores["seed"] = seed
            score_tables.append(scores)

    primary = models[0]
    frame["topic"] = assignments[0]
    assignment_path = TABLES / "topic_assignments.parquet"
    frame[["family_id", "publication_year", "publication_date", "topic"]].to_parquet(assignment_path, index=False)
    score_path = TABLES / "generative_topic_identification.csv"
    pd.concat(score_tables, ignore_index=True).to_csv(score_path, index=False)

    pairwise_ari = []
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            pairwise_ari.append(adjusted_rand_score(assignments[i], assignments[j]))
    stability = {
        "seeds": seeds,
        "pairwise_adjusted_rand_indices": pairwise_ari,
        "median_pairwise_adjusted_rand_index": float(np.median(pairwise_ari)),
        "generative_candidate_by_seed": generative_topics,
        "vision_candidate_by_seed": vision_topics,
        "generative_candidate_found_all_runs": all(bool(topics) for topics in generative_topics),
        "vision_candidate_found_all_runs": all(bool(topics) for topics in vision_topics),
        "raw_outlier_counts": raw_outlier_counts,
        "remaining_outlier_counts_after_embedding_reassignment": remaining_outlier_counts,
        "outlier_reassignment_threshold": args.reduce_outliers_threshold,
    }
    stability_path = TABLES / "topic_stability.json"
    stability_path.write_text(json.dumps(stability, indent=2), encoding="utf-8")

    topic_info_path = TABLES / "topic_information.csv"
    primary.get_topic_info().to_csv(topic_info_path, index=False)
    prevalence = annual_prevalence(frame)
    prevalence_path = TABLES / "topic_prevalence_by_year.csv"
    prevalence.to_csv(prevalence_path, index=False)

    dates = pd.to_datetime(frame["publication_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame["period_label"] = dates.dt.year.astype("Int64").astype(str) + "Q" + dates.dt.quarter.astype("Int64").astype(str)
    ordered_periods = [f"{year}Q{quarter}" for year in range(2018, 2026) for quarter in range(1, 5)]
    period_index = {period: index for index, period in enumerate(ordered_periods)}
    frame["period_index"] = frame["period_label"].map(period_index)
    quarterly_rows = []
    for target, selected_topics in (("generative_ai", generative_topics[0]), ("vision", vision_topics[0])):
        summary = (
            frame.assign(selected=frame["topic"].isin(selected_topics))
            .dropna(subset=["period_index"])
            .groupby(["period_label", "period_index"], as_index=False)
            .agg(selected_documents=("selected", "sum"), period_total=("selected", "size"))
        )
        summary["within_period_share"] = summary["selected_documents"] / summary["period_total"]
        summary["target"] = target
        quarterly_rows.append(summary)
    quarterly = pd.concat(quarterly_rows, ignore_index=True)
    quarterly_path = TABLES / "topic_prevalence_by_quarter.csv"
    quarterly.to_csv(quarterly_path, index=False)

    primary_candidate = generative_topics[0]
    test_path = TABLES / "h4_post_2023_test.json"
    figure_path = FIGURES / "generative_topic_over_time.png"
    if not primary_candidate:
        test_path.write_text(json.dumps({"supported": False, "reason": "No reproducible generative-AI topic met the prespecified term-overlap rule."}, indent=2), encoding="utf-8")
    else:
        selected = quarterly[quarterly["target"] == "generative_ai"].sort_values("period_index")
        test = {
            "generative_ai": post_2023_slope_change_test(selected, args.permutations, seeds[0]),
            "vision": post_2023_slope_change_test(
                quarterly[quarterly["target"] == "vision"].sort_values("period_index"),
                args.permutations,
                seeds[0] + 1,
            ),
            "primary_generative_topics": primary_candidate,
            "primary_vision_topics": vision_topics[0],
        }
        test_path.write_text(json.dumps(test, indent=2), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(selected["period_index"], selected["within_period_share"], marker="o")
        ax.axvline(period_index["2023Q1"], color="red", linestyle="--", linewidth=1, label="Prespecified break: 2023Q1")
        ax.set_xlabel("Publication quarter (2018Q1–2025Q4)")
        ax.set_ylabel("Within-quarter topic share")
        ax.set_title("Prevalence of the reproducibly identified generative-AI topic")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)

    model_path = RESULTS / "bertopic_primary.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(primary, handle)
    outputs = [embedding_path, embedding_metadata_path, assignment_path, score_path, stability_path, topic_info_path, prevalence_path, quarterly_path, test_path, model_path]
    if figure_path.exists():
        outputs.append(figure_path)
    write_manifest("topics", [args.input], vars(args), outputs)


if __name__ == "__main__":
    main()

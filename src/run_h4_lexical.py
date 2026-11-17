#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm

from common import DATA, FIGURES, TABLES, ensure_output_dirs, read_table, write_manifest


# Deliberately narrow expressions. Generic words such as "language", "text",
# "generation", and "transformer" are excluded because they create false positives.
PATTERNS = {
    "generative_ai": {
        "generative_ai_or_model": r"\bgenerative (?:artificial intelligence|ai|model|models|network|networks|adversarial)\b",
        "large_language_model": r"\blarge language model(?:s)?\b",
        "foundation_model": r"\bfoundation model(?:s)?\b",
        "diffusion_model": r"\bdiffusion model(?:s)?\b",
        "text_to_image": r"\btext[- ]to[- ]image\b",
        "image_generation_model": r"\bimage generation model(?:s)?\b",
        "prompt_engineering": r"\bprompt engineering\b",
        "chatgpt": r"\bchatgpt\b",
        "gpt_3_or_4": r"\bgpt[- ]?[34]\b",
    },
    "vision": {
        "computer_vision": r"\bcomputer vision\b",
        "image_processing_family": r"\bimage (?:processing|recognition|segmentation|classification)\b",
        "object_detection": r"\bobject detection\b",
        "facial_recognition": r"\bfacial recognition\b",
        "visual_recognition": r"\bvisual recognition\b",
        "target_detection": r"\btarget detection\b",
    },
}


def segmented_binomial(summary: pd.DataFrame, break_period: str = "2023Q1") -> dict:
    x_time = summary["period_index"].to_numpy(dtype=float)
    break_index = float(summary.loc[summary["period_label"] == break_period, "period_index"].iloc[0])
    post_slope = np.maximum(x_time - break_index, 0.0)
    x_linear = sm.add_constant(x_time)
    x_segmented = np.column_stack([np.ones(len(x_time)), x_time, post_slope])
    y = summary["within_period_share"].to_numpy(dtype=float)
    weights = summary["period_total"].to_numpy(dtype=float)
    linear = sm.GLM(y, x_linear, family=sm.families.Binomial(), freq_weights=weights).fit()
    segmented = sm.GLM(y, x_segmented, family=sm.families.Binomial(), freq_weights=weights).fit()
    lr = 2.0 * (segmented.llf - linear.llf)
    total_post_slope = float(segmented.params[1] + segmented.params[2])
    contrast = np.array([0.0, 1.0, 1.0])
    total_post_se = float(np.sqrt(contrast @ segmented.cov_params() @ contrast))
    total_post_z = total_post_slope / total_post_se
    return {
        "prespecified_break_period": break_period,
        "model": "binomial GLM on quarterly counts; linear time trend versus segmented time trend",
        "linear_model_log_likelihood": float(linear.llf),
        "segmented_model_log_likelihood": float(segmented.llf),
        "likelihood_ratio_chi_square": float(lr),
        "likelihood_ratio_df": 1,
        "likelihood_ratio_p": float(chi2.sf(lr, 1)),
        "pre_break_log_odds_slope": float(segmented.params[1]),
        "post_break_incremental_log_odds_slope": float(segmented.params[2]),
        "post_break_total_log_odds_slope": total_post_slope,
        "post_break_total_slope_standard_error": total_post_se,
        "post_break_total_slope_wald_p_two_sided": float(2.0 * norm.sf(abs(total_post_z))),
        "warning": "The breakpoint was prespecified. Results describe publication prevalence and do not establish a causal effect of generative AI.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA / "abstract_sample.parquet")
    parser.add_argument("--break-period", default="2023Q1")
    args = parser.parse_args()
    ensure_output_dirs()

    frame = read_table(args.input)
    required = {"family_id", "publication_year", "publication_date", "abstract_en"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["publication_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame = frame.assign(
        period_label=dates.dt.to_period("Q").astype(str),
        abstract_lower=frame["abstract_en"].fillna("").str.lower(),
    )
    ordered_periods = [f"{year}Q{quarter}" for year in range(2018, 2026) for quarter in range(1, 5)]
    period_index = {period: index for index, period in enumerate(ordered_periods)}
    frame["period_index"] = frame["period_label"].map(period_index)

    match_rows = []
    summaries = []
    annual_summaries = []
    tests = {}
    for target, named_patterns in PATTERNS.items():
        named_matches = {}
        for name, expression in named_patterns.items():
            named_matches[name] = frame["abstract_lower"].str.contains(expression, regex=True, na=False)
        match_frame = pd.DataFrame(named_matches, index=frame.index)
        selected = match_frame.any(axis=1)
        frame[target] = selected
        for name in named_patterns:
            match_rows.append({"target": target, "pattern_name": name, "documents_matched": int(match_frame[name].sum())})

        quarter = (
            frame.dropna(subset=["period_index"])
            .groupby(["period_label", "period_index"], as_index=False)
            .agg(selected_documents=(target, "sum"), period_total=(target, "size"))
            .sort_values("period_index")
        )
        quarter["within_period_share"] = quarter["selected_documents"] / quarter["period_total"]
        quarter["target"] = target
        summaries.append(quarter)
        annual = (
            frame.groupby("publication_year", as_index=False)
            .agg(selected_documents=(target, "sum"), year_total=(target, "size"))
        )
        annual["within_year_share"] = annual["selected_documents"] / annual["year_total"]
        annual["target"] = target
        annual_summaries.append(annual)
        tests[target] = segmented_binomial(quarter, args.break_period)

    tests["interpretation"] = {
        "generative_ai_increase_after_2023": bool(tests["generative_ai"]["post_break_total_log_odds_slope"] > 0),
        "vision_decline_after_2023": bool(
            tests["vision"]["post_break_total_log_odds_slope"] < 0
            and tests["vision"]["post_break_total_slope_wald_p_two_sided"] < 0.05
        ),
        "directional_pattern_consistent_with_partial_shift": bool(
            tests["generative_ai"]["post_break_total_log_odds_slope"] > 0
            and tests["vision"]["post_break_total_log_odds_slope"] < 0
            and tests["vision"]["post_break_total_slope_wald_p_two_sided"] < 0.05
        ),
        "radical_substitution_claim_supported": False,
        "note": "The indicators are non-mutually-exclusive. Parallel prevalence trends can show emergence and attenuation but cannot establish that generative AI displaced vision technologies.",
    }

    quarterly = pd.concat(summaries, ignore_index=True)
    annual = pd.concat(annual_summaries, ignore_index=True)
    quarterly_path = TABLES / "h4_lexical_prevalence_by_quarter.csv"
    annual_path = TABLES / "h4_lexical_prevalence_by_year.csv"
    patterns_path = TABLES / "h4_lexical_pattern_counts.csv"
    test_path = TABLES / "h4_lexical_break_tests.json"
    quarterly.to_csv(quarterly_path, index=False)
    annual.to_csv(annual_path, index=False)
    pd.DataFrame(match_rows).to_csv(patterns_path, index=False)
    test_path.write_text(json.dumps(tests, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    for target, display in (("generative_ai", "Generative AI (specific expressions)"), ("vision", "Vision/image processing")):
        part = quarterly[quarterly["target"] == target]
        ax.plot(part["period_index"], part["within_period_share"], marker="o", label=display)
    ax.axvline(period_index[args.break_period], color="black", linestyle="--", linewidth=1, label=f"Prespecified break: {args.break_period}")
    ax.set_xlabel("Publication quarter")
    ax.set_ylabel("Share of sampled family abstracts")
    ax.set_title("Specific lexical indicators of generative AI and vision technologies")
    ax.legend()
    fig.tight_layout()
    figure_path = FIGURES / "h4_lexical_prevalence.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    write_manifest("h4_lexical", [args.input], vars(args), [quarterly_path, annual_path, patterns_path, test_path, figure_path])


if __name__ == "__main__":
    main()

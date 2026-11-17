#!/usr/bin/env python3
"""Rebuild every number, Table 1 and Table 2 of the article from the shipped result files.

Each value is recomputed from data/ or results/tables/ and checked against the value
printed in the manuscript. The report is written to results/manuscript_numbers.md.
Exit status is non-zero if any printed value does not match.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from common import DATA, RESULTS, TABLES

checks: list[tuple[str, str, str, bool]] = []
lines: list[str] = []


def check(label: str, printed: str, value: str) -> None:
    checks.append((label, printed, value, printed == value))


def j(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


# ---------------------------------------------------------------- Section 3.1: corpus
scope = pd.read_csv(DATA / "scope_counts.csv")
families = int(scope["simple_families"].sum())
publications = int(scope["eligible_publications_before_family_deduplication"].sum())
english = families - int(scope["missing_english_abstract_families"].sum())
check("Simple patent families", "682,082", f"{families:,}")
check("Publications before family deduplication", "1,290,665", f"{publications:,}")
audit = j("data_quality_audit.json")
check("Assignee-to-assignee citation rows", "11,080,716", f"{audit['citation_edges']['rows']:,}")
check("Family-assignee-CPC rows", "6,420,968", f"{audit['assignee_cpc_panel']['rows']:,}")
check("Abstract sample size", "40,000", f"{audit['abstract_sample']['rows']:,}")
check("Families per year in the sample", "5,000",
      f"{set(audit['abstract_sample']['families_per_year'].values()).pop():,}")

# ---------------------------------------------------------------- Section 4.1 and Table 1
conc = j("network_concentration_corporate.json")
check("Eligible corporate assignees", "948", str(conc["eligible_nodes"]))
check("In-strength Gini", "0.82", f"{conc['in_strength_gini']:.2f}")
check("Top-10 share (%)", "33.7", f"{100 * conc['top_10_share']:.1f}")
null = j("h1_brokerage_null_test_corporate.json")
check("Observed maximum betweenness", "0.755", f"{null['observed_max_betweenness']:.3f}")
check("Null mean of maximum betweenness", "0.119", f"{null['null_mean']:.3f}")
check("Null-test p (upper tail)", "0.0099", f"{null['upper_tail_p']:.4f}")
check("Null randomizations", "100", str(null["repetitions"]))

cent = pd.read_csv(TABLES / "network_centrality_corporate.csv").sort_values("in_strength", ascending=False)
top10 = cent.head(10).reset_index(drop=True)
top5_pr = set(cent.nlargest(5, "pagerank")["assignee"])
names = {"IBM": "IBM", "MICROSOFT": "Microsoft", "GOOGLE": "Google",
         "SAMSUNG ELECTRONICS": "Samsung Electronics", "TENCENT TECH SHENZHEN CO LTD": "Tencent Technology",
         "INTEL CORP": "Intel Corp", "AMAZON": "Amazon", "QUALCOMM INC": "Qualcomm Inc",
         "HUAWEI TECH CO LTD": "Huawei Technologies", "BEIJING BAIDU NETCOM SCI TECH CO LTD": "Baidu"}


def position(row) -> str:
    hub = row["assignee"] in top5_pr
    broker = row["betweenness"] >= 0.25
    return "Hub and broker" if hub and broker else "Hub" if hub else "Broker" if broker else "Cited source"


table1 = pd.DataFrame({
    "Rank": range(1, 11),
    "Organization": [names.get(a, a) for a in top10["assignee"]],
    "Citation In-Strength": [f"{v:,.0f}" for v in top10["in_strength"]],
    "PageRank": [f"{v:.4f}" for v in top10["pagerank"]],
    "Betweenness": [f"{v:.4f}" for v in top10["betweenness"]],
    "Structural Position": [position(r) for _, r in top10.iterrows()],
})
printed_table1 = [
    ("IBM", "33,608", "0.0700", "0.5241", "Hub and broker"),
    ("Microsoft", "30,681", "0.0602", "0.0329", "Hub"),
    ("Google", "21,381", "0.0456", "0.0600", "Hub"),
    ("Samsung Electronics", "14,688", "0.0330", "0.3755", "Hub and broker"),
    ("Tencent Technology", "12,069", "0.0244", "0.6590", "Broker"),
    ("Intel Corp", "11,417", "0.0250", "0.0487", "Hub"),
    ("Amazon", "11,157", "0.0239", "0.0016", "Cited source"),
    ("Qualcomm Inc", "7,922", "0.0177", "0.0076", "Cited source"),
    ("Huawei Technologies", "7,880", "0.0174", "0.2917", "Broker"),
    ("Baidu", "6,713", "0.0140", "0.2872", "Broker"),
]
for (_, r), printed in zip(table1.iterrows(), printed_table1):
    check(f"Table 1 row {r['Rank']}", " | ".join(printed),
          " | ".join([r["Organization"], r["Citation In-Strength"], r["PageRank"], r["Betweenness"],
                      r["Structural Position"]]))
ibm = cent.set_index("assignee").loc["IBM"]
check("IBM PageRank (text)", "0.070", f"{ibm['pagerank']:.3f}")
check("IBM betweenness (text)", "0.524", f"{ibm['betweenness']:.3f}")
check("Highest betweenness in Table 1", "Tencent Technology (0.659)",
      f"{names[top10.loc[top10['betweenness'].idxmax(), 'assignee']]} ({top10['betweenness'].max():.3f})")
check("Microsoft betweenness", "0.033", f"{cent.set_index('assignee').loc['MICROSOFT', 'betweenness']:.3f}")
us = pd.read_csv(TABLES / "network_centrality_us_grants_corporate.csv")
check("Highest betweenness, US-grant sensitivity", "IBM", us.loc[us["betweenness"].idxmax(), "assignee"])

# ---------------------------------------------------------------- Section 4.2
pairs = pd.read_csv(TABLES / "annual_pairwise_similarity_corporate.csv")
check("Mean cosine similarity (top-30 corporate)", "0.65", f"{pairs['similarity'].mean():.2f}")
for label, beta, p in [("corporate", "0.0022", "= 0.010"),
                       ("corporate_through_2024", "0.0075", "< 0.001"),
                       ("corporate_subclass", "−0.0018", "< 0.001")]:
    h2 = j(f"h2_convergence_test_{label}.json")
    check(f"H2 trend, {label}", f"β = {beta}, p {p}",
          f"β = {h2['annual_change_in_cosine_similarity']:.4f}".replace("-", "−")
          + f", p {fmt_p(h2['permutation_p_two_sided'])}")
sel = pd.read_csv(TABLES / "cluster_selection_corporate.csv")
stab = j("cluster_stability_corporate.json")
check("Selected number of clusters", "4", str(stab["selected_k"]))
check("Silhouette at selected k", "0.396", f"{sel.set_index('k').loc[stab['selected_k'], 'silhouette']:.3f}")
check("Bootstrap ARI median / 2.5th percentile", "1.00 / 0.67",
      f"{stab['median_adjusted_rand_index']:.2f} / {stab['ari_2_5_percentile']:.2f}")
for label in ("corporate_subclass", "corporate_g06n_only"):
    check(f"Selected k, {label}", "2", str(j(f"cluster_stability_{label}.json")["selected_k"]))
clusters = pd.read_csv(TABLES / "cluster_assignments_corporate.csv")
members = clusters.groupby("cluster")["assignee"].apply(lambda s: ", ".join(sorted(s)))
lines.append("## Figure 1 cluster membership (corporate, full CPC)\n")
for c, m in members.items():
    lines.append(f"- Cluster {c}: {m}")
profiles = pd.read_csv(TABLES / "cluster_cpc_profiles_corporate.csv")
lines.append("\nMost over-represented CPC codes per cluster (location quotient, codes with share ≥ 1%):\n")
for c, g in profiles[profiles["within_cluster_share"] >= 0.01].groupby("cluster"):
    top = g.sort_values("location_quotient", ascending=False).head(5)["cpc_code"].tolist()
    lines.append(f"- Cluster {c}: {', '.join(top)}")

# ---------------------------------------------------------------- Section 4.3 and Table 2
ts = j("topic_stability.json")
check("BERTopic median pairwise ARI", "0.712", f"{ts['median_pairwise_adjusted_rand_index']:.3f}")
topics = pd.read_csv(TABLES / "topic_information.csv")
check("Number of topics (excluding outliers)", "82", str(int((topics["Topic"] >= 0).sum())))
STOP = {"the", "to", "of", "and", "a", "an", "in", "for", "is", "on", "by", "with", "first", "second"}
printed_table2 = {0: ("recommendation, user, content, product, information", "1,297"),
                  12: ("text, word, language, document, sentence", "1,079"),
                  20: ("neural, convolution, network, operation, calculation", "819"),
                  4: ("face, facial, image, recognition, expression", "669"),
                  2: ("quantum, qubit, circuit, qubits, state", "597")}
for tid, (kw, n) in printed_table2.items():
    row = topics.set_index("Topic").loc[tid]
    words = [w for w in ast.literal_eval(row["Representation"]) if w not in STOP][:5]
    check(f"Table 2 topic {tid}", f"{kw} | {n}", f"{', '.join(words)} | {int(row['Count']):,}")
lex = pd.read_csv(TABLES / "h4_lexical_prevalence_by_year.csv")
share = lex.set_index(["target", "publication_year"])["within_year_share"]
for year, printed in [(2022, "0.34"), (2023, "1.00"), (2024, "3.82"), (2025, "5.66")]:
    check(f"Generative-AI share {year} (%)", printed, f"{100 * share[('generative_ai', year)]:.2f}")
for year, printed in [(2022, "10.8"), (2025, "10.1")]:
    check(f"Vision share {year} (%)", printed, f"{100 * share[('vision', year)]:.1f}")
br = j("h4_lexical_break_tests.json")
check("Generative-AI segmented vs linear (χ², p)", "38.2, < 0.001",
      f"{br['generative_ai']['likelihood_ratio_chi_square']:.1f}, {fmt_p(br['generative_ai']['likelihood_ratio_p']).split(' ', 1)[1] if br['generative_ai']['likelihood_ratio_p'] >= 0.001 else '< 0.001'}")
check("Vision post-break slope p", "0.002", f"{br['vision']['post_break_total_slope_wald_p_two_sided']:.3f}")
check("Vision post-break slope is negative", "True", str(br["vision"]["post_break_total_log_odds_slope"] < 0))
quarter = pd.read_csv(TABLES / "h4_lexical_prevalence_by_quarter.csv")
check("Sampled families in 2023Q1", "601",
      str(int(quarter.query("period_label == '2023Q1' and target == 'vision'")["period_total"].iloc[0])))

# ---------------------------------------------------------------- report
ok = all(c[3] for c in checks)
out = ["# Numbers printed in the article, rebuilt from this package\n",
       f"All checks passed: **{ok}** ({sum(c[3] for c in checks)}/{len(checks)})\n",
       "| Item | Printed in article | Rebuilt | Match |", "|---|---|---|---|"]
out += [f"| {a} | {b} | {c} | {'yes' if d else 'NO'} |" for a, b, c, d in checks]
out += ["", "## Table 1 (rebuilt)", "", table1.to_markdown(index=False), ""] + lines
(RESULTS / "manuscript_numbers.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out[:2]))
for a, b, c, d in checks:
    if not d:
        print(f"MISMATCH {a}: printed {b!r}, rebuilt {c!r}")
raise SystemExit(0 if ok else 1)

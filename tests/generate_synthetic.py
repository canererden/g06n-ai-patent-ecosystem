#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "synthetic_data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(266665684)
firms = ["IBM CORP", "MICROSOFT CORP", "GOOGLE LLC", "SAMSUNG ELECTRONICS CO LTD"] + [f"FIRM {i}" for i in range(4, 12)]
cpcs = [f"G06N3/{i:02d}" for i in range(10)]

portfolio_rows = []
families = []
for year in range(2018, 2026):
    convergence = (year - 2018) / 7
    shared = np.linspace(2.0, 0.5, len(cpcs))
    for firm_index, firm in enumerate(firms):
        specialty = np.zeros(len(cpcs))
        specialty[firm_index % len(cpcs)] = 8.0 * (1 - convergence)
        weights = shared * (1 + convergence) + specialty + rng.uniform(0.1, 0.5, len(cpcs))
        for patent_index in range(6):
            family = f"F{year}-{firm_index}-{patent_index}"
            families.append((family, year, firm))
            chosen = rng.choice(cpcs, size=2, replace=False, p=weights / weights.sum())
            for cpc in chosen:
                portfolio_rows.append({
                    "family_id": family,
                    "publication_year": year,
                    "assignee": firm,
                    "cpc_code": cpc,
                    "fractional_family_count": 0.5,
                })

edge_rows = []
for family, year, citing in families:
    targets = rng.choice(firms, size=3, replace=False)
    for cited in targets:
        if cited != citing:
            edge_rows.append({
                "citing_family_id": family,
                "cited_family_id": f"TARGET-{cited}",
                "citing_year": year,
                "citing_assignee": citing,
                "cited_assignee": cited,
                "fractional_weight": 1.0,
            })

pd.DataFrame(portfolio_rows).to_parquet(OUT / "assignee_cpc_year.parquet", index=False)
pd.DataFrame(edge_rows).to_parquet(OUT / "citation_edges.parquet", index=False)

abstract_rows = []
templates = [
    "A neural network system trains an agent using reinforcement learning reward action policy and game environment.",
    "A semiconductor accelerator executes convolution operations using processor memory hardware circuit and tensor cores.",
    "A language model generates text from a user prompt using transformer attention semantic tokens and generative decoding.",
]
for year in range(2018, 2026):
    for topic_index, template in enumerate(templates):
        repetitions = 8 if topic_index < 2 else 2 + 2 * (year - 2018)
        for item in range(repetitions):
            abstract_rows.append({
                "family_id": f"A-{year}-{topic_index}-{item}",
                "publication_year": year,
                "abstract_en": template + f" The disclosed embodiment is example {item} for publication year {year}.",
            })
pd.DataFrame(abstract_rows).to_parquet(OUT / "abstract_sample.parquet", index=False)

#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA, TABLES, ensure_output_dirs
from run_network import harmonize


def main() -> None:
    ensure_output_dirs()
    abstract = pd.read_parquet(DATA / "abstract_sample.parquet")
    cpc = pd.read_parquet(DATA / "assignee_cpc_year.parquet")
    citation = pd.read_parquet(DATA / "citation_edges.parquet")

    parsed_dates = pd.to_datetime(abstract["publication_date"].astype(str), format="%Y%m%d", errors="coerce")
    citing = harmonize(citation["citing_assignee"])
    cited = harmonize(citation["cited_assignee"])
    audit = {
        "abstract_sample": {
            "rows": int(len(abstract)),
            "unique_families": int(abstract["family_id"].nunique()),
            "duplicate_family_rows": int(abstract.duplicated("family_id").sum()),
            "missing_abstracts": int(abstract["abstract_en"].isna().sum()),
            "unparseable_publication_dates": int(parsed_dates.isna().sum()),
            "publication_year_date_mismatches": int((parsed_dates.dt.year != abstract["publication_year"]).sum()),
            "families_per_year": {str(k): int(v) for k, v in abstract.groupby("publication_year")["family_id"].nunique().items()},
        },
        "assignee_cpc_panel": {
            "rows": int(len(cpc)),
            "unique_families": int(cpc["family_id"].nunique()),
            "unique_raw_assignees": int(cpc["assignee"].nunique()),
            "exact_duplicate_family_assignee_cpc_year_rows": int(cpc.duplicated(["family_id", "publication_year", "assignee", "cpc_code"]).sum()),
            "nonpositive_fractional_counts": int((cpc["fractional_family_count"] <= 0).sum()),
            "nonfinite_fractional_counts": int((~np.isfinite(cpc["fractional_family_count"])).sum()),
        },
        "citation_edges": {
            "rows": int(len(citation)),
            "unique_citing_families": int(citation["citing_family_id"].nunique()),
            "unique_cited_families": int(citation["cited_family_id"].nunique()),
            "exact_duplicate_expanded_rows": int(citation.duplicated().sum()),
            "nonpositive_fractional_weights": int((citation["fractional_weight"] <= 0).sum()),
            "nonfinite_fractional_weights": int((~np.isfinite(citation["fractional_weight"])).sum()),
            "self_edges_after_active_name_harmonization": int((citing == cited).sum()),
            "total_fractional_weight": float(citation["fractional_weight"].sum()),
        },
        "limitations": [
            "Name harmonization is explicit but not a comprehensive global ultimate-owner concordance.",
            "The abstract corpus is a deterministic stratified sample of 5,000 simple families per publication year.",
            "CPC assignments reflect the current public-data snapshot and can be affected by reclassification.",
        ],
    }
    out = TABLES / "data_quality_audit.json"
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

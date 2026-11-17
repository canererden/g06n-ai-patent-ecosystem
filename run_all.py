#!/usr/bin/env python3
"""Run the complete analysis sequence from the BigQuery exports.

Usage (from this folder):
    python run_all.py              # full re-analysis, including BERTopic (heavy)
    python run_all.py --available  # everything except BERTopic

The split exports in data/<name>/part-*.parquet are first joined by src/assemble_data.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / "src" / script), *arguments]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--available", action="store_true",
                        help="run only the analyses whose inputs are included in the package")
    args = parser.parse_args()

    run("assemble_data.py")
    has_edges = (DATA / "citation_edges.parquet").exists()
    has_panel = (DATA / "assignee_cpc_year.parquet").exists()
    has_abstracts = (DATA / "abstract_sample.parquet").exists()
    has_us_panel = (DATA / "assignee_cpc_year_us_grants.parquet").exists()
    has_us_edges = (DATA / "citation_edges_us_grants.parquet").exists()

    if not args.available:
        missing = [n for n, ok in [("citation_edges.parquet", has_edges),
                                   ("assignee_cpc_year.parquet", has_panel),
                                   ("abstract_sample.parquet", has_abstracts)] if not ok]
        if missing:
            raise SystemExit("Missing exports in data/: " + ", ".join(missing)
                             + "\nSee README.md.")

    if has_edges and has_panel and has_abstracts:
        run("run_data_audit.py")
    if has_edges and has_panel:
        run("run_network.py", "--organization-scope", "corporate", "--label", "corporate")
        run("run_network.py", "--organization-scope", "all", "--label", "all_assignees",
            "--bootstrap", "100", "--null", "50")
    if has_panel:
        run("run_proximity.py", "--organization-scope", "corporate", "--label", "corporate")
        run("run_proximity.py", "--organization-scope", "all", "--label", "all_assignees", "--bootstrap", "500")
        run("run_proximity.py", "--organization-scope", "all", "--label", "all_assignees_subclass",
            "--cpc-level", "subclass", "--bootstrap", "500")
        run("run_proximity.py", "--organization-scope", "corporate", "--label", "corporate_through_2024",
            "--end-year", "2024", "--bootstrap", "500")
        run("run_proximity.py", "--organization-scope", "corporate", "--label", "corporate_g06n_only",
            "--g06n-only", "--bootstrap", "500")
        run("run_proximity.py", "--organization-scope", "corporate", "--label", "corporate_subclass",
            "--cpc-level", "subclass")
        run("run_proximity.py", "--organization-scope", "corporate", "--label", "corporate_code_occurrence",
            "--weight-mode", "code_occurrence", "--bootstrap", "500")
        run("run_proximity.py", "--organization-scope", "corporate", "--label",
            "corporate_subclass_code_occurrence", "--cpc-level", "subclass",
            "--weight-mode", "code_occurrence", "--bootstrap", "500")
    if has_us_edges and has_us_panel:
        run("run_network.py", "--input", str(DATA / "citation_edges_us_grants.parquet"),
            "--portfolio-input", str(DATA / "assignee_cpc_year_us_grants.parquet"),
            "--organization-scope", "corporate", "--label", "us_grants_corporate",
            "--bootstrap", "100", "--null", "50")
    if has_us_panel:
        run("run_proximity.py", "--input", str(DATA / "assignee_cpc_year_us_grants.parquet"),
            "--organization-scope", "corporate", "--label", "us_grants_corporate", "--bootstrap", "500")
        run("run_proximity.py", "--input", str(DATA / "assignee_cpc_year_us_grants.parquet"),
            "--organization-scope", "corporate", "--label", "us_grants_corporate_subclass",
            "--cpc-level", "subclass", "--bootstrap", "500")
    if has_abstracts:
        run("run_h4_lexical.py")
        if not args.available:
            run("run_topics.py")  # computationally heavy; downloads the MiniLM embedding model
    run("make_manuscript_figures.py")
    run("reproduce_manuscript_numbers.py")


if __name__ == "__main__":
    main()

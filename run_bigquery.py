#!/usr/bin/env python3
"""Dry-run, execute, and export the official Google Patents BigQuery workflow.

Authentication uses Google Application Default Credentials. The script refuses to
execute billable SQL unless --execute and a positive --maximum-bytes-billed are both
provided. Dry-run is the default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from google.cloud import bigquery_storage


ROOT = Path(__file__).resolve().parent
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"


def render_sql(path: Path, project: str, dataset: str) -> str:
    return path.read_text(encoding="utf-8").replace("YOUR_PROJECT.YOUR_DATASET", f"{project}.{dataset}")


def dry_run(client: bigquery.Client, query: str, location: str) -> int:
    job = client.query(query, location=location, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    return int(job.total_bytes_processed or 0)


def export_query_to_parquet(client: bigquery.Client, storage_client: bigquery_storage.BigQueryReadClient, query: str, destination: Path, location: str, maximum_bytes_billed: int) -> int:
    config = bigquery.QueryJobConfig(maximum_bytes_billed=maximum_bytes_billed)
    iterator = client.query(query, location=location, job_config=config).result()
    writer = None
    rows = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        for batch in iterator.to_arrow_iterable(bqstorage_client=storage_client):
            if writer is None:
                writer = pq.ParquetWriter(destination, batch.schema, compression="zstd")
            writer.write_batch(batch)
            rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pd.DataFrame().to_parquet(destination, index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="User-owned Google Cloud billing/quota project")
    parser.add_argument("--dataset", default="pa_bus_010_analysis")
    parser.add_argument("--location", default="US")
    parser.add_argument("--maximum-bytes-billed", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    try:
        client = bigquery.Client(project=args.project, location=args.location)
    except DefaultCredentialsError as exc:
        raise SystemExit(
            "Google Application Default Credentials were not found. Authenticate first "
            "with `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS "
            "to a user-provided service-account JSON file."
        ) from exc

    scripts = [
        SQL_DIR / "01_scope_and_family_master.sql",
        SQL_DIR / "02_citation_edges.sql",
        SQL_DIR / "03_assignee_cpc_year.sql",
        SQL_DIR / "04_abstract_sample.sql",
        SQL_DIR / "05_us_grant_sensitivity.sql",
    ]
    estimates = {}
    for path in scripts:
        query = render_sql(path, args.project, args.dataset)
        try:
            estimates[path.name] = dry_run(client, query, args.location)
        except Exception as exc:  # BigQuery multi-statement DDL dry-runs can be partial.
            estimates[path.name] = {"dry_run_error": str(exc)}
    print(json.dumps({"dry_run_bytes": estimates}, indent=2))

    if not args.execute:
        print("Dry-run only. Re-run with --execute and --maximum-bytes-billed after reviewing the estimate.")
        return
    if args.maximum_bytes_billed <= 0:
        raise SystemExit("Execution refused: provide a positive --maximum-bytes-billed limit.")

    dataset = bigquery.Dataset(f"{args.project}.{args.dataset}")
    dataset.location = args.location
    client.create_dataset(dataset, exists_ok=True)
    for path in scripts:
        query = render_sql(path, args.project, args.dataset)
        config = bigquery.QueryJobConfig(maximum_bytes_billed=args.maximum_bytes_billed)
        job = client.query(query, location=args.location, job_config=config)
        job.result()
        print(f"completed: {path.name}; billed={job.total_bytes_billed or 0}; processed={job.total_bytes_processed or 0}")

    if args.skip_download:
        return
    storage_client = bigquery_storage.BigQueryReadClient()
    exports = {
        "citation_edges.parquet": f"SELECT * FROM `{args.project}.{args.dataset}.citation_edges`",
        "assignee_cpc_year.parquet": f"SELECT * FROM `{args.project}.{args.dataset}.assignee_cpc_year`",
        "abstract_sample.parquet": f"SELECT * FROM `{args.project}.{args.dataset}.abstract_sample`",
    }
    for filename, query in exports.items():
        rows = export_query_to_parquet(client, storage_client, query, DATA_DIR / filename, args.location, args.maximum_bytes_billed)
        print(f"downloaded: {filename}; rows={rows}")

    counts = client.query(
        f"""
        SELECT publication_year, COUNT(*) AS simple_families,
          COUNTIF(ARRAY_LENGTH(assignees_harmonized)=0) AS missing_assignee_families,
          COUNTIF(abstract_en IS NULL) AS missing_english_abstract_families,
          COUNT(DISTINCT country_code) AS patent_offices,
          SUM(publications_in_family) AS eligible_publications_before_family_deduplication
        FROM `{args.project}.{args.dataset}.family_master`
        GROUP BY publication_year ORDER BY publication_year
        """,
        location=args.location,
        job_config=bigquery.QueryJobConfig(maximum_bytes_billed=args.maximum_bytes_billed),
    ).result().to_dataframe(create_bqstorage_client=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(DATA_DIR / "scope_counts.csv", index=False)


if __name__ == "__main__":
    main()

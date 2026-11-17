#!/usr/bin/env python3
"""Join the split exports in data/<name>/part-*.parquet into data/<name>.parquet.

The large exports are stored in parts below 25 MB so that they can be hosted on GitHub.
Joining them in file order restores the exact row order of the BigQuery export analysed
in the article; the result is checked against data/SHA256SUMS.txt (content checksums).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common import DATA

TABLES = ["citation_edges", "assignee_cpc_year", "citation_edges_us_grants", "assignee_cpc_year_us_grants"]


def content_digest(table: pa.Table) -> str:
    """Order-sensitive checksum of the table content, independent of Parquet encoding."""
    import pandas as pd
    frame = table.to_pandas()
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def main() -> None:
    expected = {}
    sums = DATA / "SHA256SUMS.txt"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value, name = line.split(maxsplit=1)
                expected[name.strip()] = value
    for name in TABLES:
        parts = sorted((DATA / name).glob("part-*.parquet"))
        if not parts:
            print(f"skipped {name}: no parts in data/{name}/")
            continue
        table = pa.concat_tables([pq.read_table(p) for p in parts])
        pq.write_table(table, DATA / f"{name}.parquet", compression="zstd")
        digest = content_digest(table)
        key = f"content:{name}"
        status = "" if key not in expected else (" (checksum OK)" if expected[key] == digest else " (CHECKSUM MISMATCH)")
        print(f"{name}.parquet: {table.num_rows:,} rows from {len(parts)} parts{status}")


if __name__ == "__main__":
    main()

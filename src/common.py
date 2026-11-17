from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"


def ensure_output_dirs() -> None:
    for path in (RESULTS, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(name: str, inputs: Iterable[Path], parameters: dict, outputs: Iterable[Path]) -> None:
    ensure_output_dirs()
    def display_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(ROOT.resolve()))
        except ValueError:
            return str(resolved)

    payload = {
        "analysis": name,
        "inputs": [{"path": display_path(p), "sha256": sha256(p)} for p in inputs],
        "parameters": parameters,
        "outputs": [{"path": display_path(p), "sha256": sha256(p)} for p in outputs],
    }
    (RESULTS / f"{name}_manifest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.allclose(x.sum(), 0):
        return float("nan")
    x = np.sort(np.clip(x, 0, None))
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def hhi(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if np.allclose(x.sum(), 0):
        return float("nan")
    shares = x / x.sum()
    return float(np.square(shares).sum())

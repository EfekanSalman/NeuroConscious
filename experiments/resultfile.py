"""Format of experiment result files (kept free of heavy imports so analysis tools stay light)."""

import json
from pathlib import Path

SCHEMA_VERSION = 1


def load_results(path: Path) -> dict:
    """Reads a result file written by experiments.run and checks its schema version."""
    results = json.loads(Path(path).read_text())
    if results.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema {results.get('schema')!r} (expected {SCHEMA_VERSION})")
    return results

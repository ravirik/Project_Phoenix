"""Confines any externally-supplied data path to the project's `data/` directory."""
from __future__ import annotations

import os

# Resolved once relative to the project's working directory.
DATA_DIR = os.path.realpath(os.getenv("PHOENIX_DATA_DIR", "data"))

def safe_data_path(candidate: str | None, default: str = "data/default_ingestion.parquet") -> str:
    """Returns `candidate` if it resolves inside DATA_DIR, else `default`."""
    if not candidate:
        return default
    try:
        resolved = os.path.realpath(os.path.join(DATA_DIR, candidate)
                                    if not os.path.isabs(candidate) else candidate)
    except Exception:
        return default

    try:
        if os.path.commonpath([resolved, DATA_DIR]) != DATA_DIR:
            return default
    except ValueError:
        return default

    return resolved

def is_safe_data_path(candidate: str | None) -> bool:
    if not candidate:
        return False
    sentinel = "__REJECTED__"
    return safe_data_path(candidate, default=sentinel) != sentinel
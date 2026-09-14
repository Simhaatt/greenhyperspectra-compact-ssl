"""Project path resolution.

Every script in this repository resolves its data and results directories
through this module.  The project root is, in order of precedence:

1. the ``HSI_ROOT`` environment variable, if set;
2. the parent directory of ``src/`` (i.e. the repository checkout).

Nothing is hard-coded to a particular machine, so the pipeline runs
unchanged on Linux, macOS and Windows.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_ROOT = ROOT / "results"
FIGURES_ROOT = ROOT / "figures"

__all__ = ["ROOT", "DATA_DIR", "RAW_DIR", "RESULTS_ROOT", "FIGURES_ROOT"]

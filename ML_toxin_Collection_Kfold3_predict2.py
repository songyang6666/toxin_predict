"""Compatibility entry point for the cleaned toxin prediction workflow.

The original analysis script with this name was refactored into reusable
modules under `src/ml_toxin_predict/` and the runnable script
`run_toxin_predict.py`. This wrapper preserves the legacy file name for
manuscript traceability while keeping the executable workflow maintainable.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).resolve().parent / "run_toxin_predict.py",
        run_name="__main__",
    )

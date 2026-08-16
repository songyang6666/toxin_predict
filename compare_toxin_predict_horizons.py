"""Compatibility entry point for comparing toxin prediction horizons."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ml_toxin_predict.horizons import main  # noqa: E402


if __name__ == "__main__":
    main()

"""Entry point for reviewer-requested non-interpolated weekly validation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ml_toxin_predict.weekly_validation import main  # noqa: E402


if __name__ == "__main__":
    main()

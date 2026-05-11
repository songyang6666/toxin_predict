from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = PROJECT_ROOT.parent / "toxin_predict_github_release.zip"

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
}
EXCLUDED_NAMES = {
    ".DS_Store",
}
EXCLUDED_PREFIXES = {
    "data/outputs/",
    "data/processed/",
}


def should_include(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    relative_text = relative.as_posix()
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    if any(relative_text.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return path.name == ".gitkeep"
    if path.suffix in {".pyc", ".pyo", ".zip", ".tar", ".gz"}:
        return False
    return True


def main() -> None:
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with ZipFile(ARCHIVE_PATH, "w", ZIP_DEFLATED) as archive:
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if path.is_file() and should_include(path):
                archive.write(path, Path(PROJECT_ROOT.name) / path.relative_to(PROJECT_ROOT))
    print(ARCHIVE_PATH)


if __name__ == "__main__":
    main()

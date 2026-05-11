from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_WQ_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

DEFAULT_WORKBOOK = RAW_WQ_DIR / "western_lake_erie_water_quality_2012_2022.xlsx"

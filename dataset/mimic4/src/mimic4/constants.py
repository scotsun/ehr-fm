from pathlib import Path

DATA_PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = DATA_PROJECT_ROOT / "data/raw_mimic4"
EXPORT_DIR = DATA_PROJECT_ROOT / "data/mimic4_tokens.parquet"

if __name__ == "__main__":
    print(f"Project root: {DATA_PROJECT_ROOT}")
    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Export: {EXPORT_DIR}")


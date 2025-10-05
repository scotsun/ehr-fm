from pathlib import Path

DATA_PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPORT_DIR = DATA_PROJECT_ROOT / "data/instacart.parquet"

if __name__ == "__main__":
    print(DATA_PROJECT_ROOT)

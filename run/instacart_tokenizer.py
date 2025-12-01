import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer import get_tokenizer


def main():
    config = {
        "vocab_size": 45000,
        "tokenizer_path": "./dataset/instacart/data/tk.json",
        "min_frequency": 5,
        "data_folder": "/hpc/group/engelhardlab/ms1008/instacart/instacart.parquet",
        "metadata_file_path": "/hpc/group/engelhardlab/ms1008/instacart/metadata.csv",
        "max_worker": 10,
        "token_col": "product_id",
    }
    config["length"] = len(os.listdir(config["data_folder"]))
    get_tokenizer(config)


if __name__ == "__main__":
    main()

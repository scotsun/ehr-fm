import pandas as pd
from src.tokenizer import get_tokenizer


def main():
    datasets = []
    for i in range(2):
        data_sample = pd.read_csv(
            "./dataset/EHRSHOT_ASSETS/data/ehrshot.csv",
            nrows=50000,
            dtype={"patient_id": str, "visit_id": str},
        )
        data_sample.drop("Unnamed: 0", axis=1, inplace=True)
        data_sample.sort_values(by=["patient_id", "visit_id", "start"], inplace=True)
        datasets.append(data_sample)
    get_tokenizer(
        datasets,
        {
            "tokenizer_path": "./test.json",
            "patient_id_col": "patient_id",
            "token_col": "code",
            "min_frequency": 1,
        },
    )


if __name__ == "__main__":
    main()

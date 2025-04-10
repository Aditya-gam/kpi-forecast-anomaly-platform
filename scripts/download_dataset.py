# scripts/download_dataset.py
import os
import requests


def download_dataset(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    download_dataset(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx",
        save_path="data/raw/online_retail_II.xlsx"
    )

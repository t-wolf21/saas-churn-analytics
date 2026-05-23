from pathlib import Path
import zipfile
import subprocess

DATASET = "rivalytics/saas-subscription-and-churn-analytics-dataset"

RAW_DATA_DIR = Path("data/raw")
ZIP_PATH = RAW_DATA_DIR / "saas-subscription-and-churn-analytics-dataset.zip"

def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(RAW_DATA_DIR),
        ],
        check=True,
    )

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

    ZIP_PATH.unlink()

    print("Dataset downloaded and extracted.")


if __name__ == "__main__":
    main()
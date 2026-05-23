from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
SAMPLE_DATA_DIR = Path("data/sample")


def load_csv(file_name: str, sample: bool = False) -> pd.DataFrame:
    data_dir = SAMPLE_DATA_DIR if sample else RAW_DATA_DIR
    path = data_dir / file_name

    if not path.exists():
        raise FileNotFoundError(f"File {file_name} not found in {data_dir}")

    return pd.read_csv(path)

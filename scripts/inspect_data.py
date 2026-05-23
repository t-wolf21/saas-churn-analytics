from pathlib import Path
import pandas as pd

def inspect_csv(path: Path) -> None:
    df = pd.read_csv(path)
    print(df.head())

def main() -> None:
    DATA_DIR = Path("data/raw")
    csv_files = DATA_DIR.glob("*.csv")

    for csv_file in csv_files:
        inspect_csv(csv_file)

if __name__ == "__main__":
    main()
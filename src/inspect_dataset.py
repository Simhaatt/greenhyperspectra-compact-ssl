import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"


def main() -> None:
    for path in sorted(RAW_DIR.glob("greenhyperspectra_labeled*.parquet")):
        df = pd.read_parquet(path)
        wave_cols = [col for col in df.columns if str(col).isdigit()]
        print(f"\n{path.name}")
        print(f"shape: {df.shape}")
        print(f"wavelengths: {len(wave_cols)} ({wave_cols[0]}-{wave_cols[-1]})")
        print(f"metadata columns: {[col for col in df.columns if not str(col).isdigit()]}")

        trait_cols = [col for col in ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"] if col in df.columns]
        if trait_cols:
            missing = df[trait_cols].isna().sum().sort_values(ascending=False)
            available = df[trait_cols].notna().sum().sort_values(ascending=False)
            print("available trait labels:")
            for trait in trait_cols:
                print(f"  {trait}: {available[trait]} available, {missing[trait]} missing")


if __name__ == "__main__":
    main()

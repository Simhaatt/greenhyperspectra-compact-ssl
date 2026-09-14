from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "vegetation_index_baseline"
PAPER_TABLE_DIR = ROOT / "results" / "paper_tables"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [str(col) for col in df.columns if str(col).isdigit()]


def nearest_wave(waves: list[str], value: int) -> str:
    return min(waves, key=lambda wave: abs(int(wave) - value))


def normalized_difference(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    numerator = df[a].astype(float) - df[b].astype(float)
    denominator = df[a].astype(float) + df[b].astype(float)
    return numerator / denominator.replace(0, np.nan)


def build_index_frame(df: pd.DataFrame, waves: list[str]) -> pd.DataFrame:
    w = {value: nearest_wave(waves, value) for value in [531, 550, 570, 670, 680, 705, 720, 750, 800, 850, 970, 1240, 1450, 1600]}
    out = pd.DataFrame(index=df.index)
    out["NDVI_800_680"] = normalized_difference(df, w[800], w[680])
    out["GNDVI_800_550"] = normalized_difference(df, w[800], w[550])
    out["NDRE_750_705"] = normalized_difference(df, w[750], w[705])
    out["red_edge_nd_720_670"] = normalized_difference(df, w[720], w[670])
    out["PRI_531_570"] = normalized_difference(df, w[531], w[570])
    out["NDWI_970_850"] = normalized_difference(df, w[850], w[970])
    out["NDWI_1240_850"] = normalized_difference(df, w[850], w[1240])
    out["MSI_1600_800"] = df[w[1600]].astype(float) / df[w[800]].replace(0, np.nan).astype(float)
    out["water_ratio_1450_970"] = df[w[1450]].astype(float) / df[w[970]].replace(0, np.nan).astype(float)
    out["simple_ratio_800_680"] = df[w[800]].astype(float) / df[w[680]].replace(0, np.nan).astype(float)
    out["red_edge_slope_750_705"] = (df[w[750]].astype(float) - df[w[705]].astype(float)) / (int(w[750]) - int(w[705]))
    out["green_red_ratio_550_680"] = df[w[550]].astype(float) / df[w[680]].replace(0, np.nan).astype(float)
    out["nir_swir_ratio_800_1600"] = df[w[800]].astype(float) / df[w[1600]].replace(0, np.nan).astype(float)
    out["swir_water_nd_1600_1240"] = normalized_difference(df, w[1600], w[1240])
    out["cci_like_720_700_680"] = (df[w[720]].astype(float) - df[w[705]].astype(float)) / (
        df[w[705]].astype(float) - df[w[680]].astype(float)
    ).replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = rmse(y_true, y_pred)
    if value == 0:
        return float("inf")
    return float(np.std(y_true, ddof=1) / value)


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, pred),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "rpd": rpd(y_true, pred),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    waves = wavelength_columns(train)
    x_train_all = build_index_frame(train, waves)
    x_test_all = build_index_frame(test, waves)

    models = {
        "VI_Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 25))),
        "VI_RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=0,
            n_jobs=-1,
        ),
    }

    rows: list[dict[str, object]] = []
    for target in TRAITS:
        train_mask = train[target].notna() & x_train_all.notna().all(axis=1)
        test_mask = test[target].notna() & x_test_all.notna().all(axis=1)
        x_train = x_train_all.loc[train_mask].to_numpy(dtype=np.float32)
        y_train = train.loc[train_mask, target].to_numpy(dtype=np.float32)
        x_test = x_test_all.loc[test_mask].to_numpy(dtype=np.float32)
        y_test = test.loc[test_mask, target].to_numpy(dtype=np.float32)
        print(f"Trait={target} train={len(y_train)} test={len(y_test)}")
        for name, model in models.items():
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            row = {
                "target": target,
                "model": name,
                "n_indices": x_train.shape[1],
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                **metric_row(y_test, pred),
            }
            rows.append(row)
            print(f"  {name}: R2={row['r2']:.3f} RMSE={row['rmse']:.6f}")

    results = pd.DataFrame(rows).sort_values(["target", "model"])
    leaderboard = (
        results.groupby("model", as_index=False)
        .agg(mean_r2=("r2", "mean"), median_r2=("r2", "median"), mean_rpd=("rpd", "mean"))
        .sort_values("mean_r2", ascending=False)
    )
    results.to_csv(RESULTS_DIR / "vegetation_index_baseline_results.csv", index=False)
    leaderboard.to_csv(RESULTS_DIR / "vegetation_index_baseline_leaderboard.csv", index=False)
    results.to_csv(PAPER_TABLE_DIR / "table59_vegetation_index_baseline.csv", index=False)
    leaderboard.to_csv(PAPER_TABLE_DIR / "table59b_vegetation_index_baseline_leaderboard.csv", index=False)
    print("\nLeaderboard:")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

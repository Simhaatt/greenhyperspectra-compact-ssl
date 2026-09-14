from __future__ import annotations
import os

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "band_selection"
RANDOM_STATE = 42
BUDGETS = [5, 10, 20, 30]
RANDOM_REPEATS = 20


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).isdigit()]


def load_split(target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    waves = wavelength_columns(train)

    train = train.dropna(subset=[target]).reset_index(drop=True)
    test = test.dropna(subset=[target]).reset_index(drop=True)

    train = train[np.isfinite(train[waves + [target]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    test = test[np.isfinite(test[waves + [target]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    return train, test, waves


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    score_rmse = rmse(y_true, y_pred)
    if score_rmse == 0:
        return float("inf")
    return float(np.std(y_true, ddof=1) / score_rmse)


def ridge_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]
    )


def plsr_model(n_bands: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", PLSRegression(n_components=min(20, n_bands))),
        ]
    )


def score_model(
    target: str,
    model_name: str,
    selector_name: str,
    budget: int,
    selected_idx: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    waves: list[str],
) -> dict[str, object]:
    models = {
        "Ridge": ridge_model(),
        "PLSR": plsr_model(len(selected_idx)),
    }
    model = models[model_name]

    start = perf_counter()
    model.fit(x_train[:, selected_idx], y_train)
    train_seconds = perf_counter() - start

    start = perf_counter()
    pred = np.asarray(model.predict(x_test[:, selected_idx])).reshape(-1)
    predict_seconds = perf_counter() - start

    selected_wavelengths = [waves[i] for i in selected_idx]
    return {
        "selector": selector_name,
        "model": model_name,
        "target": target,
        "budget": budget,
        "bands": len(selected_idx),
        "rmse": rmse(y_test, pred),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "rpd": rpd(y_test, pred),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "selected_wavelengths": ",".join(selected_wavelengths),
    }


def top_by_abs_correlation(x_train: np.ndarray, y_train: np.ndarray, budget: int) -> np.ndarray:
    y_centered = y_train - y_train.mean()
    x_centered = x_train - x_train.mean(axis=0, keepdims=True)
    numerator = np.abs(x_centered.T @ y_centered)
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
    corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.argsort(corr)[-budget:][::-1]


def top_by_mutual_information(x_train: np.ndarray, y_train: np.ndarray, budget: int) -> np.ndarray:
    mi = mutual_info_regression(x_train, y_train, random_state=RANDOM_STATE, n_neighbors=5)
    return np.argsort(mi)[-budget:][::-1]


def summarize_random(target: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: list[dict[str, object]] = []
    df = pd.DataFrame(rows)
    metric_cols = ["rmse", "mae", "r2", "rpd", "train_seconds", "predict_seconds"]
    for (selector, model, budget), group in df.groupby(["selector", "model", "budget"], sort=False):
        row: dict[str, object] = {
            "selector": selector,
            "model": model,
            "target": target,
            "budget": int(budget),
            "bands": int(budget),
            "selected_wavelengths": "random_repeat_mean",
        }
        for metric in metric_cols:
            values = group[metric].astype(float)
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        grouped.append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train compact-band baselines for one GreenHyperSpectra trait.")
    parser.add_argument("--target", default="cab", choices=["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"])
    args = parser.parse_args()
    target = args.target

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train, test, waves = load_split(target)
    x_train = train[waves].to_numpy(dtype=np.float32)
    y_train = train[target].to_numpy(dtype=np.float32)
    x_test = test[waves].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    print(f"Target: {target}")
    print(f"Train rows: {len(y_train)} | Test rows: {len(y_test)} | Full bands: {len(waves)}")

    rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    print("Scoring full-spectrum references...")
    full_idx = np.arange(len(waves))
    for model_name in ["Ridge", "PLSR"]:
        row = score_model(target, model_name, "FullSpectrum", len(waves), full_idx, x_train, y_train, x_test, y_test, waves)
        rows.append(row)

    rng = np.random.default_rng(RANDOM_STATE)
    print("Computing mutual information ranking once...")
    mi_ranking = np.argsort(mutual_info_regression(x_train, y_train, random_state=RANDOM_STATE, n_neighbors=5))[::-1]

    for budget in BUDGETS:
        print(f"\nBudget {budget}")
        selectors = {
            "AbsCorr": top_by_abs_correlation(x_train, y_train, budget),
            "MutualInfo": mi_ranking[:budget],
        }

        for selector_name, selected_idx in selectors.items():
            selected_idx = np.asarray(selected_idx, dtype=int)
            selected_rows.append(
                {
                    "selector": selector_name,
                    "budget": budget,
                    "selected_wavelengths": ",".join(waves[i] for i in selected_idx),
                }
            )
            for model_name in ["Ridge", "PLSR"]:
                row = score_model(target, model_name, selector_name, budget, selected_idx, x_train, y_train, x_test, y_test, waves)
                rows.append(row)
                print(f"{selector_name}+{model_name}: RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}")

        for repeat in range(RANDOM_REPEATS):
            selected_idx = rng.choice(len(waves), size=budget, replace=False)
            for model_name in ["Ridge", "PLSR"]:
                row = score_model(target, model_name, "Random", budget, selected_idx, x_train, y_train, x_test, y_test, waves)
                row["repeat"] = repeat
                random_rows.append(row)

    rows.extend(summarize_random(target, random_rows))

    results = pd.DataFrame(rows).sort_values(["budget", "rmse"], ascending=[True, True])
    random_detail = pd.DataFrame(random_rows)
    selected = pd.DataFrame(selected_rows)

    results.to_csv(RESULTS_DIR / f"{target}_band_selection_results.csv", index=False)
    random_detail.to_csv(RESULTS_DIR / f"{target}_random_band_selection_repeats.csv", index=False)
    selected.to_csv(RESULTS_DIR / f"{target}_selected_wavelengths.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / f"{target}_band_selection_results.csv")
    print(RESULTS_DIR / f"{target}_random_band_selection_repeats.csv")
    print(RESULTS_DIR / f"{target}_selected_wavelengths.csv")
    print("\nTop compact-band results:")
    compact = results[results["selector"] != "FullSpectrum"].sort_values("rmse").head(12)
    print(compact[["selector", "model", "budget", "rmse", "mae", "r2", "rpd"]].to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations
import os

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_cab_kan_band_selection import TrainConfig, fit_predict_kan

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
BEAM_DIR = ROOT / "results" / "beam"
RESULTS_DIR = ROOT / "results" / "compact_regressors"
TRAITS = ["cab", "cw", "cm"]
RANDOM_STATE = 42


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
    value = rmse(y_true, y_pred)
    if value == 0:
        return float("inf")
    return float(np.std(y_true, ddof=1) / value)


def load_beam_indices(target: str, budget: int, waves: list[str]) -> tuple[np.ndarray, str]:
    beam = pd.read_csv(BEAM_DIR / f"{target}_beam_band_selection_results.csv")
    row = beam[beam["budget"] == budget].sort_values("rmse").iloc[0]
    selected_waves = str(row["selected_wavelengths"]).split(",")
    wave_to_idx = {wave: idx for idx, wave in enumerate(waves)}
    selected_idx = np.asarray([wave_to_idx[wave] for wave in selected_waves], dtype=int)
    return selected_idx, ",".join(selected_waves)


def evaluate_model(
    target: str,
    name: str,
    model: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected_wavelengths: str,
) -> tuple[dict[str, object], np.ndarray]:
    start = perf_counter()
    model.fit(x_train, y_train)
    train_seconds = perf_counter() - start

    start = perf_counter()
    pred = np.asarray(model.predict(x_test)).reshape(-1)
    predict_seconds = perf_counter() - start

    row: dict[str, object] = {
        "target": target,
        "selector": "BeamSearch",
        "model": name,
        "budget": x_train.shape[1],
        "bands": x_train.shape[1],
        "rmse": rmse(y_test, pred),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "rpd": rpd(y_test, pred),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "selected_wavelengths": selected_wavelengths,
    }
    return row, pred


def model_list(budget: int) -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = [
        (
            "PLSR",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression(n_components=min(20, budget))),
                ]
            ),
        ),
        (
            "Ridge",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
                ]
            ),
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "MLP",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            alpha=0.001,
                            batch_size=128,
                            learning_rate_init=0.001,
                            max_iter=600,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=40,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]

    if XGBRegressor is not None:
        models.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=700,
                    max_depth=3,
                    learning_rate=0.02,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )
    return models


def run_trait(target: str, budget: int, skip_kan: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test, waves = load_split(target)
    selected_idx, selected_wavelengths = load_beam_indices(target, budget, waves)

    x_train_all = train[waves].to_numpy(dtype=np.float32)
    y_train = train[target].to_numpy(dtype=np.float32)
    x_test_all = test[waves].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    x_train = x_train_all[:, selected_idx]
    x_test = x_test_all[:, selected_idx]

    rows: list[dict[str, object]] = []
    predictions = pd.DataFrame({"target": target, "y_true": y_test})

    print(f"\nTrait={target} | train={len(y_train)} test={len(y_test)} bands={budget}")
    print(f"Selected wavelengths: {selected_wavelengths}")

    for name, model in model_list(budget):
        print(f"Training {target} BeamSearch+{name}...")
        row, pred = evaluate_model(target, name, model, x_train, y_train, x_test, y_test, selected_wavelengths)
        rows.append(row)
        predictions[name] = pred
        print(f"  RMSE={row['rmse']:.6f}, R2={row['r2']:.4f}")

    if not skip_kan:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = TrainConfig(max_epochs=500, patience=60)
        print(f"Training {target} BeamSearch+KAN on {device}...")
        metrics, pred = fit_predict_kan(
            x_train_all,
            y_train,
            x_test_all,
            y_test,
            selected_idx,
            device,
            config,
            RANDOM_STATE + budget,
        )
        row: dict[str, object] = {
            "target": target,
            "selector": "BeamSearch",
            "model": "KAN",
            "budget": budget,
            "bands": budget,
            "device": str(device),
            "selected_wavelengths": selected_wavelengths,
            **metrics,
        }
        rows.append(row)
        predictions["KAN"] = pred
        print(f"  RMSE={row['rmse']:.6f}, R2={row['r2']:.4f}")

    return pd.DataFrame(rows), predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nonlinear regressors on BeamSearch-selected compact wavelengths.")
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--skip-kan", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for trait in args.traits:
        results, predictions = run_trait(trait, args.budget, args.skip_kan)
        result_frames.append(results)
        prediction_frames.append(predictions)

    all_results = pd.concat(result_frames, ignore_index=True).sort_values(["target", "rmse"])
    all_predictions = pd.concat(prediction_frames, ignore_index=True)

    all_results.to_csv(RESULTS_DIR / f"beam_k{args.budget}_compact_regressor_results.csv", index=False)
    all_predictions.to_csv(RESULTS_DIR / f"beam_k{args.budget}_compact_regressor_predictions.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / f"beam_k{args.budget}_compact_regressor_results.csv")
    print(RESULTS_DIR / f"beam_k{args.budget}_compact_regressor_predictions.csv")
    print("\nResults:")
    print(all_results[["target", "model", "budget", "rmse", "mae", "r2", "rpd"]].to_string(index=False))


if __name__ == "__main__":
    main()
